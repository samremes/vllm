# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.model_executor.layers.quantization.online.mxfp4 import Mxfp4OnlineLinearMethod
from vllm.platforms.rocm import on_gfx950

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.linear import KimiRoutedInputLinear

logger = init_logger(__name__)


_TP_ASM_CONTEXT_POOL: dict[tuple[Any, ...], Any] = {}


def _get_or_create_tp_asm_context(
    tp_group,
    device: torch.device,
    max_tokens: int,
    hidden_dim: int,
    topk: int,
    num_experts: int,
    inter_dim: int,
):
    from aiter.ops.asm.fmoe_tp_sdma import TpAsmContext

    key = (tuple(tp_group.ranks), hidden_dim, num_experts, topk, max_tokens)
    ctx = _TP_ASM_CONTEXT_POOL.get(key)
    if ctx is None or getattr(ctx, "_closed", True):
        ctx = TpAsmContext(
            tp_group.cpu_group,
            device,
            max_tokens,
            hidden_dim,
            topk,
            num_experts,
            inter_dim,
        )
        _TP_ASM_CONTEXT_POOL[key] = ctx
    return ctx


class ROCmLatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Mirrors CUDA's LatentMoERunner, but currently only the up projection
    -sharded path is implemented. (Tier 2)

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        transform = self.routed_output_transform
        up_proj = getattr(transform, "up_proj", None)
        tp_size = get_tensor_model_parallel_world_size()

        self._up_proj_preshard = bool(getattr(transform, "row_sharded", False))
        self._up_proj_shard_size = 0
        self._tail_shardable = (
            up_proj is not None
            and tp_size > 1
            and (self._up_proj_preshard or up_proj.weight.shape[0] % tp_size == 0)
            and self._shared_experts is not None
            and not self.moe_config.is_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )
        if self._tail_shardable:
            assert up_proj is not None
            self._up_proj_shard_size = (
                up_proj.weight.shape[0]
                if self._up_proj_preshard
                else up_proj.weight.shape[0] // tp_size
            )
        else:
            logger.warning_once(
                "K3 latent-MoE tail is not shardable under this config, "
                "falling back to the replicated up-projection.",
                scope="global",
            )
        self._logged_sharded_tail = False
        self._quantized_routed_input: bool | None = None
        self._fmoe_tp_sdma_enabled: bool | None = None
        self._fmoe_tp_sdma_epi_mode: str = "rmsnorm_only"
        self._fmoe_tp_sdma_ctx: Any | None = None
        self._fmoe_tp_sdma_min_tokens = max(
            1, int(os.getenv("VLLM_ROCM_FMOE_TP_SDMA_MIN_TOKENS", "256"))
        )

    def _init_fmoe_tp_sdma(self) -> None:
        self._fmoe_tp_sdma_enabled = False
        if not (
            envs.VLLM_ROCM_USE_AITER
            and envs.VLLM_ROCM_USE_AITER_MOE
            and envs.VLLM_ROCM_USE_AITER_FMOE_TP_SDMA
            and on_gfx950()
        ):
            return
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.lora_config is not None:
            return
        moe_cfg = self.moe_config
        p_cfg = moe_cfg.moe_parallel_config
        if (
            p_cfg.tp_size not in (2, 4, 8)
            or p_cfg.dp_size != 1
            or p_cfg.ep_size != 1
            or p_cfg.pcp_size != 1
            or moe_cfg.is_sequence_parallel
            or moe_cfg.hidden_dim != 3584
            or moe_cfg.hidden_dim_unpadded != 3584
            or moe_cfg.intermediate_size_per_partition != 384
            or moe_cfg.intermediate_size_per_partition_unpadded != 384
            or moe_cfg.num_experts != 896
            or moe_cfg.experts_per_token != 16
            or moe_cfg.activation_situ_beta != 4.0
            or moe_cfg.activation_situ_linear_beta != 25.0
            or moe_cfg.has_bias
            or self.gate is not None
            or self.shared_expert_gate is not None
            or self.routed_experts.activation != MoEActivation.SITU
            or self.routed_experts.apply_router_weight_on_input
            or self.routed_scaling_factor != 1.0
            or not rocm_aiter_ops.is_fused_moe_situv2_enabled()
            or rocm_aiter_ops.is_fused_moe_situv2_gate_up_interleaved()
        ):
            return
        kernel = getattr(self._quant_method, "moe_kernel", None)
        if (
            kernel is None
            or not isinstance(kernel.fused_experts, AiterMxfp4Experts)
            or kernel.fused_experts.post_gemm2_hook is not None
        ):
            return
        w13 = getattr(self.routed_experts, "w13_weight", None)
        w2 = getattr(self.routed_experts, "w2_weight", None)
        w13_scale = getattr(self.routed_experts, "w13_weight_scale", None)
        w2_scale = getattr(self.routed_experts, "w2_weight_scale", None)
        if (
            w13 is None
            or w2 is None
            or w13_scale is None
            or w2_scale is None
            or not getattr(w13, "is_shuffled", False)
            or getattr(w13, "gate_up_interleave", False)
        ):
            return
        transform = self.routed_output_transform
        if transform is None:
            return
        if (
            transform.norm is not None
            and transform.norm.weight.dtype == torch.bfloat16
            and abs(float(transform.norm.variance_epsilon) - 1e-6) <= 1e-12
        ):
            epi_mode = "rmsnorm_only"
        else:
            epi_mode = "raw_bf16"
        try:
            from aiter.ops.asm import fmoe_tp_sdma

            reg = fmoe_tp_sdma.load_fmoe_tp_sdma_registry()
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning_once(
                "fmoe_tp_sdma unavailable, using stock MoE path: %s",
                exc,
                scope="global",
            )
            return
        key = (p_cfg.tp_size, 3584, 384, 16, 64, "fp8", "situ", epi_mode, 0)
        if key not in reg:
            return
        max_batched = 8192
        if vllm_config is not None and vllm_config.scheduler_config is not None:
            max_batched = int(
                vllm_config.scheduler_config.max_num_batched_tokens or 8192
            )
        max_tokens = min(32768, max(max_batched, 8192))
        self._fmoe_tp_sdma_ctx = _get_or_create_tp_asm_context(
            get_tp_group(),
            w13.device,
            max_tokens,
            3584,
            16,
            896,
            384,
        )
        self._fmoe_tp_sdma_epi_mode = epi_mode
        self._fmoe_tp_sdma_enabled = True
        logger.info_once(
            "Kimi-K3 prefill using fmoe_tp_sdma "
            "(epi_mode=%s, min_tokens=%d, max_tokens=%d)",
            epi_mode,
            self._fmoe_tp_sdma_min_tokens,
            max_tokens,
            scope="global",
        )

    def _use_fmoe_tp_sdma(self, num_tokens: int) -> bool:
        return (
            bool(self._fmoe_tp_sdma_enabled)
            and num_tokens >= self._fmoe_tp_sdma_min_tokens
        )

    def _run_fmoe_tp_sdma(
        self,
        x_in: torch.Tensor,
        x_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        from aiter.ops.asm.fmoe_tp_sdma import TpMoeCall, tp_moe_layer_asm

        ctx = self._fmoe_tp_sdma_ctx
        assert ctx is not None
        epi_mode = self._fmoe_tp_sdma_epi_mode

        w1 = self.routed_experts.w13_weight.data.contiguous()
        w2 = self.routed_experts.w2_weight.data.contiguous()
        w1_scale = self.routed_experts.w13_weight_scale.data.contiguous()
        w2_scale = self.routed_experts.w2_weight_scale.data.contiguous()
        if epi_mode == "rmsnorm_only":
            transform = self.routed_output_transform
            assert transform is not None and transform.norm is not None
            norm_weight: torch.Tensor | None = transform.norm.weight.data.contiguous()
            eps = float(transform.norm.variance_epsilon)
        else:
            norm_weight = None
            eps = 1e-6
        topk_ids_i32 = topk_ids.to(torch.int32).contiguous()
        topk_weights_f32 = topk_weights.to(torch.float32).contiguous()
        x_c = x_in.contiguous()
        xs_c = x_scale.contiguous() if x_scale is not None else None

        tokens = int(x_c.shape[0])
        if tokens <= ctx.max_tokens:
            call = TpMoeCall(
                x=x_c,
                topk_ids=topk_ids_i32,
                topk_weights=topk_weights_f32,
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                norm_weight=norm_weight,
                eps=eps,
                x_scale=xs_c,
            )
            return tp_moe_layer_asm(
                call,
                ctx,
                epi_mode=epi_mode,
                a_dtype="fp8",
                act="situ",
            ).out

        out = torch.empty(
            (tokens, ctx.model_dim), dtype=torch.bfloat16, device=x_c.device
        )
        for start in range(0, tokens, ctx.max_tokens):
            end = min(start + ctx.max_tokens, tokens)
            call = TpMoeCall(
                x=x_c[start:end],
                topk_ids=topk_ids_i32[start:end],
                topk_weights=topk_weights_f32[start:end],
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                norm_weight=norm_weight,
                eps=eps,
                x_scale=xs_c[start:end] if xs_c is not None else None,
            )
            tp_moe_layer_asm(
                call,
                ctx,
                epi_mode=epi_mode,
                out=out[start:end],
                a_dtype="fp8",
                act="situ",
            )
        return out

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
        fused_state: Literal["unreduced", "reduced", "normed"] = "unreduced",
    ) -> torch.Tensor:
        """Tier 2: column-parallel up-projection folded into the final reduce."""
        if not self._logged_sharded_tail:
            self._logged_sharded_tail = True
            logger.info_once(
                "Kimi-K3 latent-MoE tail: up-projecting only this rank's "
                "hidden shard into the shared output.",
                scope="global",
            )

        transform = self.routed_output_transform
        assert transform is not None

        if fused_state == "normed":
            latent = fused_output
        else:
            latent = (
                fused_output
                if fused_state == "reduced"
                else tensor_model_parallel_all_reduce(fused_output)
            )
            if transform.norm is not None:
                latent = transform.norm(latent)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        weight = transform.up_proj.weight
        up_proj_shard = (
            weight
            if self._up_proj_preshard
            else weight.narrow(0, shard_start, shard_size)
        )
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._tail_shardable and not self._fused_output_is_reduced:
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ):
        if self._use_fmoe_tp_sdma(hidden_states.shape[0]):
            self.routed_experts._ensure_moe_quant_config_init()
            assert self._shared_experts is not None and shared_experts_input is not None
            shared_output = self._shared_experts._layer(shared_experts_input)
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=torch.int32,
                input_ids=input_ids,
            )
            if hidden_states.shape[-1] != self.moe_config.hidden_dim:
                assert self.routed_input_transform is not None
                x_in, x_scale = self.routed_input_transform.forward_quantized_fp8(
                    hidden_states
                )
            else:
                x_in, x_scale = hidden_states, None
            latent_out = self._run_fmoe_tp_sdma(x_in, x_scale, topk_ids, topk_weights)
            return shared_output, latent_out

        return super()._forward_impl(
            hidden_states, router_logits, shared_experts_input, input_ids
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # The modular kernel is selected after weight loading. Resolve its
        # compatible prepare adapter once, during warmup before graph capture.
        if self._quantized_routed_input is None:
            self._quantized_routed_input = enable_quantized_routed_input(self)
        if self._fmoe_tp_sdma_enabled is None:
            self._init_fmoe_tp_sdma()

        use_fmoe_tp_sdma = self._use_fmoe_tp_sdma(hidden_states.shape[0])
        transform = self.routed_input_transform
        defer_projection = (
            (self._quantized_routed_input or use_fmoe_tp_sdma)
            and shared_experts_input is None
            and transform is not None
            and getattr(transform, "_token_group", None) is not None
            and hidden_states.shape[0] >= transform.min_sharded_tokens
            and hidden_states.dtype == torch.bfloat16
        )
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if defer_projection:
            shared_experts_input = hidden_states
            og_hidden_dim_pre_xform: int | None = self.moe_config.hidden_dim
            og_hidden_dim_post_xform: int | None = hidden_states.shape[-1]
        elif shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        if not defer_projection:
            hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
                self._maybe_pad_hidden_states(
                    shared_experts_input,
                    hidden_states,
                )
            )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if defer_projection or self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = cast(tuple[torch.Tensor, torch.Tensor], result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        fused_state: Literal["unreduced", "reduced", "normed"] = "unreduced"
        if use_fmoe_tp_sdma:
            fused_state = (
                "normed" if self._fmoe_tp_sdma_epi_mode == "rmsnorm_only" else "reduced"
            )

        result = self._shard_up_proj_tail(
            fused_output,
            shared_output,
            og_hidden_dim_post_xform,
            fused_state=fused_state,
        )

        return self._maybe_add_zero_expert_output(result)


class KimiLatentAiterExperts(AiterMxfp4Experts):
    def output_hidden_dim(self, hidden_dim: int) -> int:
        # Input may still be in model space when prepare owns the projection.
        return self.moe_config.hidden_dim

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        experts, rows, intermediate, width, topk = super().moe_problem_size(
            a1, w1, w2, topk_ids
        )
        if a1.dtype == torch.float4_e2m1fn_x2:
            width *= 2
        return experts, rows, intermediate, width, topk


class KimiLatentPrepare(MoEPrepareAndFinalizeNoDPEPModular):
    def __init__(self, projection: "KimiRoutedInputLinear"):
        self.projection = projection

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        if a1.shape[-1] == self.projection.input_size:
            assert a1.shape[0] >= self.projection.min_sharded_tokens
            assert not apply_router_weight_on_input
            values, scales = self.projection.forward_quantized(a1)
            return values, scales, None, None, None
        return super().prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )


def enable_quantized_routed_input(runner) -> bool:
    """Use the existing modular prepare/scale interface for Kimi A4 prefill."""
    # linear.py imports this module, so import the layer class here.
    from vllm.models.kimi_k3.amd.linear import KimiRoutedInputLinear

    projection = runner.routed_input_transform
    if (
        not isinstance(projection, KimiRoutedInputLinear)
        or projection._token_group is None
        # These methods return BF16 activations. Their existing A4 quantization
        # can happen on the local token shard before gathering values/scales.
        or (
            projection.weight.dtype != torch.bfloat16
            and not isinstance(
                projection.quant_method,
                (Mxfp4OnlineLinearMethod, Fp8PtpcOnlineLinearMethod),
            )
        )
        or get_current_vllm_config().lora_config is not None
        or runner.routed_experts.activation != MoEActivation.SITU
        or not rocm_aiter_ops.is_fused_moe_situv2_enabled()
        or rocm_aiter_ops.is_fused_moe_situv2_gate_up_interleaved()
        or runner.routed_experts.apply_router_weight_on_input
    ):
        return False
    method = runner._quant_method
    kernel = getattr(method, "moe_kernel", None)
    if (
        kernel is None
        or type(kernel.fused_experts) is not AiterMxfp4Experts
        or type(kernel.prepare_finalize) is not MoEPrepareAndFinalizeNoDPEPModular
        or kernel.fused_experts.post_gemm2_hook is not None
        or runner.moe_config.hidden_dim != projection.output_size
        or runner.moe_config.hidden_dim_unpadded != projection.output_size
    ):
        return False
    method.moe_kernel = FusedMoEKernel(
        KimiLatentPrepare(projection),
        KimiLatentAiterExperts(
            kernel.fused_experts.moe_config, kernel.fused_experts.quant_config
        ),
    )
    return True
