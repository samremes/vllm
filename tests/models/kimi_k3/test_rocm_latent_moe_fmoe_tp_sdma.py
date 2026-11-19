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
        return all(
            (2, 3584, 384, 16, 64, a_dt, "situ", epi, 0) in reg
            for a_dt in ("fp4", "fp8")
            for epi in ("rmsnorm_only", "raw_bf16", "raw_bf16_sp")
        )
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
    if moe.routed_output_transform.row_sharded:
        shard_size = up_w.shape[0]
        shard_start = tp_rank * shard_size
        up_w_shard = up_w
    else:
        shard_size = up_w.shape[0] // dist.get_world_size()
        shard_start = tp_rank * shard_size
        up_w_shard = up_w.narrow(0, shard_start, shard_size)
    out = shared_out.clone()
    out[:, shard_start : shard_start + shard_size].addmm_(latent, up_w_shard.t())
    out_f32 = out.float()
    dist.all_reduce(out_f32)
    return out_f32.to(torch.bfloat16)


def _init_weights(module: torch.nn.Module, seed: int, tp_rank: int) -> None:
    g_rep = torch.Generator(device="cuda").manual_seed(seed)
    g_tp = torch.Generator(device="cuda").manual_seed(seed + 1000 + tp_rank)
    up_is_tp = bool(
        getattr(getattr(module, "routed_output_transform", None), "row_sharded", False)
    )
    tp_keys = ["shared_experts", "w13_weight", "w2_weight"]
    if up_is_tp:
        tp_keys.append("routed_expert_up_proj")
    for name, p in module.named_parameters():
        is_tp = any(k in name for k in tp_keys)
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
    up_proj_sp: bool,
) -> dict[tuple[str, int], float]:
    import vllm.envs as envs
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.kimi_k3.amd.linear import KimiMoE

    prev_sp_env = os.environ.get("VLLM_ROCM_KIMI_UP_PROJ_SP")
    os.environ["VLLM_ROCM_KIMI_UP_PROJ_SP"] = "1" if up_proj_sp else "0"
    try:
        return _run_suite_inner(
            rank,
            device,
            rms_norm_eps,
            expected_epi_mode,
            up_proj_sp,
            envs,
            set_current_vllm_config,
            get_tp_group,
            set_forward_context,
            process_weights_after_loading,
            KimiMoE,
        )
    finally:
        if prev_sp_env is None:
            os.environ.pop("VLLM_ROCM_KIMI_UP_PROJ_SP", None)
        else:
            os.environ["VLLM_ROCM_KIMI_UP_PROJ_SP"] = prev_sp_env


def _run_suite_inner(
    rank: int,
    device: torch.device,
    rms_norm_eps: float,
    expected_epi_mode: str,
    up_proj_sp: bool,
    envs,
    set_current_vllm_config,
    get_tp_group,
    set_forward_context,
    process_weights_after_loading,
    KimiMoE,
) -> dict[tuple[str, int], float]:
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

        assert moe.routed_output_transform is not None
        assert moe.routed_output_transform.row_sharded is (not up_proj_sp)

        _init_weights(moe, 101, rank)
        routed = moe.experts.routed_experts
        w13_raw = routed.w13_weight.data.clone()
        w13_s_raw = routed.w13_weight_scale.data.clone()
        w2_raw = routed.w2_weight.data.clone()
        w2_s_raw = routed.w2_weight_scale.data.clone()

        process_weights_after_loading(moe, vllm_config.model_config, device)
        runner = moe.experts

        # Verify DCP=1 enable_token_sharding() on TP2
        down_proj = moe.routed_expert_down_proj
        assert down_proj is not None
        down_proj.enable_token_sharding()
        assert down_proj._token_group is get_tp_group()
        assert down_proj.min_sharded_tokens == 2048

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

        g_in = torch.Generator(device="cuda").manual_seed(2026)

        def _make_x(t: int) -> torch.Tensor:
            return (
                torch.randn(
                    t, 7168, dtype=torch.bfloat16, generator=g_in, device="cuda"
                )
                * 0.25
            )

        sqnr_by_dtype: dict[tuple[str, int], float] = {}
        prev_adt = os.environ.get("VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE")
        try:
            for a_dtype in ("fp4", "fp8"):
                os.environ["VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE"] = a_dtype
                runner._init_fmoe_tp_sdma()
                assert runner._fmoe_tp_sdma_enabled is True
                assert runner._fmoe_tp_sdma_sp_enabled is up_proj_sp
                assert runner._fmoe_tp_sdma_a_dtype == a_dtype
                assert runner._fmoe_tp_sdma_epi_mode == expected_epi_mode

                # Accuracy across decode fallback (T=16), non-SP (T=256),
                # aligned SP (T=512, 2048, 4096), and ragged SP (T=533).
                for t in (16, 256, 512, 533, 2048, 4096):
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
                        assert not runner._use_fmoe_tp_sdma_sp(t)
                        assert _cos_sim(y_sdma, y_stock) > 0.99
                    else:
                        assert runner._use_fmoe_tp_sdma(t)
                        if up_proj_sp and t >= runner._fmoe_tp_sdma_sp_min_tokens:
                            assert runner._use_fmoe_tp_sdma_sp(t)
                        else:
                            assert not runner._use_fmoe_tp_sdma_sp(t)
                        cos_sdma = _cos_sim(y_sdma, y_ref)
                        sqnr_sdma = _sqnr_db(y_sdma, y_ref)
                        sqnr_stock = _sqnr_db(y_stock, y_ref)
                        sqnr_by_dtype[(a_dtype, t)] = sqnr_sdma
                        if a_dtype == "fp4":
                            cos_stock = _cos_sim(y_sdma, y_stock)
                            assert cos_stock > 0.99, (
                                f"Low cos(sdma, stock) (fp4) at T={t}: {cos_stock}"
                            )
                            assert cos_sdma > 0.965, (
                                f"Low cos(sdma, ref) (fp4) at T={t}: {cos_sdma}"
                            )
                        else:
                            assert cos_sdma > 0.98, (
                                f"Low cos_sim (fp8) at T={t}: {cos_sdma}"
                            )
                            assert sqnr_sdma > sqnr_stock, (
                                f"Expected fp8 SQNR ({sqnr_sdma:.2f}) > "
                                f"stock ({sqnr_stock:.2f})"
                            )
                            assert sqnr_sdma > sqnr_by_dtype[("fp4", t)], (
                                f"Expected fp8 SQNR ({sqnr_sdma:.2f}) > "
                                f"fp4 ({sqnr_by_dtype[('fp4', t)]:.2f})"
                            )

                # Exercise distributed forward_quantized / forward_quantized_fp8
                x2 = _make_x(2048)
                with torch.no_grad():
                    if a_dtype == "fp4":
                        q_in, s_in = down_proj.forward_quantized(x2)
                        assert q_in.shape == (2048, 3584 // 2)
                    else:
                        q_in, s_in = down_proj.forward_quantized_fp8(x2)
                        assert q_in.shape == (2048, 3584)
                    assert s_in.shape == (2048, 3584 // 32)
        finally:
            if prev_adt is None:
                os.environ.pop("VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE", None)
            else:
                os.environ["VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE"] = prev_adt

        # torch.compile safety across T=16, 256, 1024, 4096 for both fp4 and fp8
        prev_adt = os.environ.get("VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE")
        try:
            for a_dtype in ("fp4", "fp8"):
                os.environ["VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE"] = a_dtype
                runner._init_fmoe_tp_sdma()
                compiled_moe = torch.compile(moe, backend="inductor", dynamic=True)
                with torch.no_grad():
                    for t in (16, 256, 1024, 4096):
                        x = _make_x(t)
                        y_comp = compiled_moe(x)
                        y_eager = moe(x)
                        y_eager2 = moe(x)
                        rel = _rel_rms(y_comp, y_eager)
                        rel_rep = _rel_rms(y_eager, y_eager2)
                        cos = _cos_sim(y_comp, y_eager)
                        tol = max(1e-2, rel_rep * 1.5 + 1e-2)
                        assert rel < tol and cos > 0.985, (
                            f"torch.compile ({a_dtype}) mismatch at T={t}: "
                            f"rel={rel}, rel_rep={rel_rep}, cos={cos}"
                        )
                del compiled_moe
        finally:
            if prev_adt is None:
                os.environ.pop("VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE", None)
            else:
                os.environ["VLLM_ROCM_FMOE_TP_SDMA_A_DTYPE"] = prev_adt

        del moe, w13_raw, w13_s_raw, w2_raw, w2_s_raw
        torch.cuda.empty_cache()
        return sqnr_by_dtype


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
    os.environ["VLLM_ROCM_KIMI_UP_PROJ_SP_MIN_TOKENS"] = "512"

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

    sqnr_non_sp = _run_suite(rank, device, 1e-5, "raw_bf16", False)
    sqnr_sp = _run_suite(rank, device, 1e-5, "raw_bf16", True)
    for a_dt in ("fp4", "fp8"):
        for t in (512, 533, 2048, 4096):
            assert sqnr_sp[(a_dt, t)] > sqnr_non_sp[(a_dt, t)], (
                f"Expected raw_bf16_sp SQNR ({sqnr_sp[(a_dt, t)]:.2f}) > "
                f"non-SP raw_bf16 ({sqnr_non_sp[(a_dt, t)]:.2f}) for {a_dt} T={t}"
            )
    _run_suite(rank, device, 1e-6, "rmsnorm_only", False)
    _run_suite(rank, device, 1e-6, "rmsnorm_only", True)


def test_kimi_k3_latent_moe_fmoe_tp_sdma_tp2() -> None:
    port = _find_free_port()
    mp.spawn(_tp2_worker, args=(port,), nprocs=2, join=True)
