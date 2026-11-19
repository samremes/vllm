# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2 tests for Kimi-K3 ROCm LatentMoE fmoe_tp_sdma integration."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from vllm.platforms import current_platform


def _has_fmoe_tp_sdma_tp2() -> bool:
    if not current_platform.is_rocm() or torch.cuda.device_count() < 2:
        return False
    try:
        from vllm._aiter_ops import is_aiter_found_and_supported
        from vllm.platforms.rocm import on_gfx950

        if not (is_aiter_found_and_supported() and on_gfx950()):
            return False
        from aiter.ops.asm import fmoe_tp_sdma

        reg = fmoe_tp_sdma.load_fmoe_tp_sdma_registry()
        return (
            2,
            3584,
            384,
            16,
            64,
            "fp8",
            "situ",
            "rmsnorm_only",
            0,
        ) in reg and (
            2,
            3584,
            384,
            16,
            64,
            "fp8",
            "situ",
            "raw_bf16",
            0,
        ) in reg
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_fmoe_tp_sdma_tp2(),
    reason="Requires >=2 gfx950 ROCm GPUs and aiter.ops.asm.fmoe_tp_sdma TP2 kernels",
)

_FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).pow(2).mean().sqrt()
    denom = b.float().pow(2).mean().sqrt().clamp_min(1e-12)
    return float((diff / denom).item())


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    )


def _sqnr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    sig = b.float().pow(2).mean().clamp_min(1e-24)
    err = (a.float() - b.float()).pow(2).mean().clamp_min(1e-24)
    return float((10.0 * torch.log10(sig / err)).item())


def _dequant_mxfp4(w_packed_u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    lut = _FP4_LUT.to(w_packed_u8.device)
    lo = w_packed_u8 & 0x0F
    hi = (w_packed_u8 >> 4) & 0x0F
    vals = torch.stack((lut[lo.long()], lut[hi.long()]), dim=-1).reshape(
        *w_packed_u8.shape[:-1], -1
    )
    exp = scale_u8.to(torch.int32) - 127
    scale = torch.pow(2.0, exp.float())
    vals = vals.reshape(*vals.shape[:-1], -1, 32) * scale.unsqueeze(-1)
    return vals.reshape(*w_packed_u8.shape[:-1], -1)


def _situ_fp32(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    a = (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5))
    return a * (25.0 * torch.tanh(u * 0.04))


def _bf16_ref_forward(
    x: torch.Tensor,
    moe,
    w13_u8: torch.Tensor,
    w13_s_u8: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_s_u8: torch.Tensor,
    tp_rank: int,
) -> torch.Tensor:
    import torch.distributed as dist

    tokens = x.shape[0]
    logits = moe.gate(x)[0].float()
    scores = torch.sigmoid(logits)
    bias = moe.gate.e_score_correction_bias.float()
    _, topk_ids = torch.topk(scores + bias, 16, dim=-1, sorted=True)
    topk_w = scores.gather(-1, topk_ids)
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)

    shared_out = moe.shared_experts(x)
    latent_in = moe.routed_expert_down_proj(x)[0]

    routed_local = torch.zeros_like(latent_in, dtype=torch.float32)
    for start in range(0, tokens, 256):
        end = min(start + 256, tokens)
        x_chunk = latent_in[start:end].float()
        ids_c = topk_ids[start:end]
        w_c = topk_w[start:end]
        acc = torch.zeros_like(x_chunk)
        for e in ids_c.unique().tolist():
            mask = ids_c == e
            t_idx, k_idx = mask.nonzero(as_tuple=True)
            w13_e = _dequant_mxfp4(w13_u8[e], w13_s_u8[e])
            w2_e = _dequant_mxfp4(w2_u8[e], w2_s_u8[e])
            gu = F.linear(x_chunk[t_idx], w13_e)
            g, u = gu.chunk(2, dim=-1)
            out_e = F.linear(_situ_fp32(g, u), w2_e)
            acc.index_add_(0, t_idx, out_e * w_c[t_idx, k_idx].unsqueeze(-1))
        routed_local[start:end] = acc

    dist.all_reduce(routed_local)
    latent = routed_local.to(torch.bfloat16)
    if moe.routed_expert_norm is not None:
        latent = moe.routed_expert_norm(latent)

    up_w = moe.routed_expert_up_proj.weight
    shard_size = up_w.shape[0]
    shard_start = tp_rank * shard_size
    out = shared_out.clone()
    out[:, shard_start : shard_start + shard_size].addmm_(latent, up_w.t())
    out_f32 = out.float()
    dist.all_reduce(out_f32)
    return out_f32.to(torch.bfloat16)


def _init_weights(module: torch.nn.Module, seed: int, tp_rank: int) -> None:
    g_rep = torch.Generator(device="cuda").manual_seed(seed)
    g_tp = torch.Generator(device="cuda").manual_seed(seed + 1000 + tp_rank)
    for name, p in module.named_parameters():
        is_tp = any(
            k in name
            for k in (
                "shared_experts",
                "routed_expert_up_proj",
                "w13_weight",
                "w2_weight",
            )
        )
        g = g_tp if is_tp else g_rep
        if p.dtype in (torch.bfloat16, torch.float32):
            if "norm" in name:
                p.data.fill_(1.0)
            elif "bias" in name:
                p.data.copy_(torch.randn(p.shape, generator=g, device="cuda") * 1e-3)
            else:
                v = torch.randn(p.shape, generator=g, device="cuda")
                p.data.copy_(v * (p.shape[-1] ** -0.5))
        elif p.dtype == torch.uint8 and "scale" in name:
            p.data.copy_(torch.randint(123, 127, p.shape, generator=g, device="cuda"))
        elif p.dtype == torch.uint8:
            p.data.copy_(torch.randint(0, 256, p.shape, generator=g, device="cuda"))
        else:
            raise TypeError(f"Unexpected parameter dtype {name}: {p.dtype}")


def _make_vllm_config(tp_size: int, rms_norm_eps: float):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    hf_cfg = KimiLinearConfig(
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=384 * tp_size,
        num_experts=896,
        num_experts_per_token=16,
        num_shared_experts=1,
        routed_expert_hidden_size=3584,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        latent_moe_use_norm=True,
        rms_norm_eps=rms_norm_eps,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
        use_grouped_topk=False,
        dsa_config={
            "dsa_layers": [4],
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_num_heads": 32,
            "index_rope_dim": 0,
        },
    )
    quant_cfg = Mxfp4Config()
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
            tokenizer="moonshotai/Kimi-Linear-48B-A3B-Instruct",
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            seed=0,
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
        scheduler_config=SchedulerConfig(
            max_num_seqs=128,
            max_num_batched_tokens=16384,
            max_model_len=65536,
            is_encoder_decoder=False,
        ),
        quant_config=quant_cfg,
    )
    vllm_config.model_config.hf_config = hf_cfg
    vllm_config.model_config.hf_text_config = hf_cfg
    return hf_cfg, quant_cfg, vllm_config


def _run_suite(
    rank: int,
    device: torch.device,
    rms_norm_eps: float,
    expected_epi_mode: str,
) -> None:
    import vllm.envs as envs
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.kimi_k3.amd.linear import KimiMoE

    hf_cfg, quant_cfg, vllm_config = _make_vllm_config(2, rms_norm_eps)

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(device):
                moe = KimiMoE(
                    hf_cfg,
                    quant_config=quant_cfg,
                    prefix="model.layers.1.mlp",
                    layer_idx=1,
                )
        finally:
            torch.set_default_dtype(prev_dtype)

        _init_weights(moe, 101, rank)
        routed = moe.experts.routed_experts
        w13_raw = routed.w13_weight.data.clone()
        w13_s_raw = routed.w13_weight_scale.data.clone()
        w2_raw = routed.w2_weight.data.clone()
        w2_s_raw = routed.w2_weight_scale.data.clone()

        process_weights_after_loading(moe, vllm_config.model_config, device)
        runner = moe.experts
        runner._init_fmoe_tp_sdma()

        # Eligibility checks
        assert runner._fmoe_tp_sdma_enabled is True
        assert runner._fmoe_tp_sdma_epi_mode == expected_epi_mode

        prev_env = os.environ.get("VLLM_ROCM_USE_AITER_FMOE_TP_SDMA")
        try:
            os.environ["VLLM_ROCM_USE_AITER_FMOE_TP_SDMA"] = "0"
            assert not envs.VLLM_ROCM_USE_AITER_FMOE_TP_SDMA
            runner._init_fmoe_tp_sdma()
            assert runner._fmoe_tp_sdma_enabled is False
        finally:
            if prev_env is None:
                os.environ.pop("VLLM_ROCM_USE_AITER_FMOE_TP_SDMA", None)
            else:
                os.environ["VLLM_ROCM_USE_AITER_FMOE_TP_SDMA"] = prev_env
        runner._init_fmoe_tp_sdma()
        assert runner._fmoe_tp_sdma_enabled is True
        assert runner._fmoe_tp_sdma_epi_mode == expected_epi_mode

        g_in = torch.Generator(device="cuda").manual_seed(2026)

        def _make_x(t: int) -> torch.Tensor:
            return (
                torch.randn(
                    t, 7168, dtype=torch.bfloat16, generator=g_in, device="cuda"
                )
                * 0.25
            )

        # Accuracy across decode fallback (T=16) and prefill (T=256..8192)
        for t in (16, 256, 4096, 8192):
            x = _make_x(t)
            with torch.no_grad():
                runner._fmoe_tp_sdma_enabled = False
                y_stock = moe(x)

                runner._fmoe_tp_sdma_enabled = True
                y_sdma = moe(x)

                y_ref = _bf16_ref_forward(
                    x, moe, w13_raw, w13_s_raw, w2_raw, w2_s_raw, rank
                )

            if t < runner._fmoe_tp_sdma_min_tokens:
                assert not runner._use_fmoe_tp_sdma(t)
                assert _cos_sim(y_sdma, y_stock) > 0.99
            else:
                assert runner._use_fmoe_tp_sdma(t)
                cos_sdma = _cos_sim(y_sdma, y_ref)
                sqnr_sdma = _sqnr_db(y_sdma, y_ref)
                sqnr_stock = _sqnr_db(y_stock, y_ref)
                assert cos_sdma > 0.98, f"Low cos_sim at T={t}: {cos_sdma}"
                assert sqnr_sdma > sqnr_stock, (
                    f"Expected fmoe_tp_sdma SQNR ({sqnr_sdma:.2f}) > "
                    f"stock ({sqnr_stock:.2f})"
                )

        # Exercise distributed forward_quantized_fp8 + defer_projection at T=8192
        from vllm.distributed import get_tp_group

        down_proj = moe.routed_expert_down_proj
        assert down_proj is not None
        prev_tok_group = down_proj._token_group
        try:
            down_proj._token_group = get_tp_group()
            x8 = _make_x(8192)
            with torch.no_grad():
                q_fp8, s_e8m0 = down_proj.forward_quantized_fp8(x8)
                assert q_fp8.shape == (8192, 3584)
                assert s_e8m0.shape == (8192, 3584 // 32)
                y_ref8 = _bf16_ref_forward(
                    x8, moe, w13_raw, w13_s_raw, w2_raw, w2_s_raw, rank
                )
                runner._fmoe_tp_sdma_enabled = False
                y_stock_sharded = moe(x8)
                runner._fmoe_tp_sdma_enabled = True
                y_sdma_sharded = moe(x8)
                assert _cos_sim(y_sdma_sharded, y_ref8) > 0.98
                assert _sqnr_db(y_sdma_sharded, y_ref8) > _sqnr_db(
                    y_stock_sharded, y_ref8
                )
        finally:
            down_proj._token_group = prev_tok_group

        # torch.compile safety across T=16, 256, 4096
        compiled_moe = torch.compile(moe, backend="inductor", dynamic=True)
        with torch.no_grad():
            for t in (16, 256, 4096):
                x = _make_x(t)
                y_comp = compiled_moe(x)
                y_eager = moe(x)
                y_eager2 = moe(x)
                rel = _rel_rms(y_comp, y_eager)
                rel_rep = _rel_rms(y_eager, y_eager2)
                cos = _cos_sim(y_comp, y_eager)
                tol = max(1e-2, rel_rep * 1.5 + 1e-2)
                assert rel < tol and cos > 0.985, (
                    f"torch.compile mismatch at T={t}: "
                    f"rel={rel}, rel_rep={rel_rep}, cos={cos}"
                )

        del compiled_moe, moe, w13_raw, w13_s_raw, w2_raw, w2_s_raw
        torch.cuda.empty_cache()


def _tp2_worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    os.environ["VLLM_ROCM_USE_AITER_MOE"] = "1"
    os.environ["VLLM_ROCM_USE_AITER_FMOE_TP_SDMA"] = "1"

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    _, _, init_cfg = _make_vllm_config(2, 1e-5)
    with set_current_vllm_config(init_cfg):
        init_workspace_manager(device)
        init_distributed_environment(
            world_size=2,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=2)

    for rms_norm_eps, expected_epi_mode in (
        (1e-6, "rmsnorm_only"),
        (1e-5, "raw_bf16"),
    ):
        _run_suite(rank, device, rms_norm_eps, expected_epi_mode)


def test_kimi_k3_latent_moe_fmoe_tp_sdma_tp2() -> None:
    port = _find_free_port()
    mp.spawn(_tp2_worker, args=(port,), nprocs=2, join=True)
