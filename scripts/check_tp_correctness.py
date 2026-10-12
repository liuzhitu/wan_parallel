"""One torchrun entry point for TP1/TP2 layer and full-DiT correctness checks.

Numerical discrepancies are recorded, never raised: inspect summary.status.
Exit code zero means the suite completed, not that every check passed.
Execution errors (loading, CUDA, NCCL) propagate with an incomplete JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F

from parallel.tp_linear import ColumnParallelLinear, RowParallelLinear
from parallel.tp_weight_loader import load_linear_weight
from runtime.distributed import destroy_distributed_environment, init_distributed_environment
from runtime.parallel_state import (
    destroy_model_parallel, get_tensor_model_parallel_group, initialize_model_parallel,
)


class Report:
    def __init__(self, path: Path, rank: int, metadata: dict) -> None:
        self.path = path
        self.rank = rank
        self.data = dict(complete=False, metadata=metadata, checks=[])
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def add(self, name: str, status: str, **details) -> None:
        self.data["checks"].append(dict(name=name, status=status, **details))
        self.save()
        print(f"[rank {self.rank}] {status} {name} "
              + json.dumps(details, ensure_ascii=False), flush=True)

    def invariant(self, name: str, passed: bool, **details) -> None:
        self.add(name, "PASS" if passed else "FAIL", **details)

    def compare(self, name: str, actual: torch.Tensor, expected: torch.Tensor,
                *, rtol: float = 0, atol: float = 0, exact: bool = False) -> None:
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            self.add(name, "FAIL", actual_shape=list(actual.shape), expected_shape=list(expected.shape),
                     actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype))
            return
        if not (torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item()):
            self.add(name, "FAIL", reason="nonfinite output or reference")
            return
        a, b = actual.float(), expected.float()
        delta = a - b
        max_abs = delta.abs().max().item()
        norm = torch.linalg.vector_norm(b).item()
        delta_norm = torch.linalg.vector_norm(delta).item()
        relative_l2 = delta_norm / norm if norm else (0.0 if delta_norm == 0 else None)
        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        bad = (~close).sum().item()
        # REVIEW means the full-reference precision criterion is NOT met. It
        # does not accept the error as harmless or silently enlarge tolerances.
        status = "PASS" if bad == 0 else ("FAIL" if exact else "REVIEW")
        self.add(name, status, max_abs=max_abs, relative_l2=relative_l2,
                 max_reference=b.abs().max().item(), bad_elements=int(bad), elements=a.numel(),
                 rtol=rtol, atol=atol, dtype=str(actual.dtype))


def shared_random(shape, device, group, seed: int) -> torch.Tensor:
    value = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    dist.broadcast(value, src=0, group=group)
    return value


def rank_agreement(report: Report, name: str, value: torch.Tensor, group) -> None:
    source = value.detach().clone()
    dist.broadcast(source, src=0, group=group)
    report.compare(name, value, source, exact=True)


@torch.no_grad()
def linear_cases(report: Report, path: Path, label: str, device, group) -> None:
    rank = dist.get_rank(group)
    tp_size = dist.get_world_size(group)
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        w1, b1, w2, b2 = [checkpoint.get_tensor(f"blocks.0.ffn.{name}").to(device=device, dtype=torch.float32)
                          for name in ("0.weight", "0.bias", "2.weight", "2.bias")]
    hidden, dim = w1.shape
    width = hidden // tp_size
    left, right = rank * width, (rank + 1) * width
    col = ColumnParallelLinear(dim, hidden, group, device, torch.float32)
    row = RowParallelLinear(hidden, dim, group, device, torch.float32)
    identities = [id(p) for module in (col, row) for p in module.parameters()]
    load_linear_weight(col, path, "blocks.0.ffn.0")
    load_linear_weight(row, path, "blocks.0.ffn.2")
    report.invariant(f"linear/{label}/parameter_identity",
                     identities == [id(p) for module in (col, row) for p in module.parameters()])
    for name, value, expected in (
        ("column.weight", col.weight, w1[left:right]), ("column.bias", col.bias, b1[left:right]),
        ("row.weight", row.weight, w2[:, left:right]), ("row.bias", row.bias, b2),
    ):
        report.compare(f"linear/{label}/load/{name}", value, expected, exact=True)

    def same_order(x, ffn):
        partials = []
        for shard in range(tp_size):
            a, b = shard * width, (shard + 1) * width
            z = F.gelu(F.linear(x, w1[a:b].contiguous(), b1[a:b].contiguous()), approximate="tanh") \
                if ffn else x[..., a:b].contiguous()
            partials.append(F.linear(z, w2[:, a:b].contiguous(), bias=b2 if shard == 0 else None))
        output = partials[0]
        for partial in partials[1:]:
            output = output + partial
        return output

    for bf16 in (False, True):
        mode = "bf16" if bf16 else "fp32"
        rtol, atol = (0.02, 0.02) if bf16 else (1e-4, 1e-5)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            for tokens in (1, 17, 128):
                name = f"linear/{label}/{mode}/tokens={tokens}"
                x = shared_random((1, tokens, dim), device, group, 42 + tokens)
                full_col = F.linear(x, w1, b1)
                local = col(x)
                report.compare(name + "/column", local, full_col[..., left:right], rtol=rtol, atol=atol)
                z = shared_random((1, tokens, hidden), device, group, 142 + tokens)
                actual_row = row(z[..., left:right].contiguous())
                actual_ffn = row(F.gelu(local, approximate="tanh"))
                report.compare(name + "/row", actual_row, F.linear(z, w2, b2), rtol=rtol, atol=atol)
                report.compare(name + "/ffn", actual_ffn,
                               F.linear(F.gelu(full_col, approximate="tanh"), w2, b2), rtol=rtol, atol=atol)
                if bf16:
                    report.compare(name + "/row_same_order", actual_row, same_order(z, False), exact=True)
                    report.compare(name + "/ffn_same_order", actual_ffn, same_order(x, True), exact=True)
                rank_agreement(report, name + "/row_rank_agreement", actual_row, group)
                rank_agreement(report, name + "/ffn_rank_agreement", actual_ffn, group)
            zero_col = col(torch.zeros((1, 1, dim), device=device))
            zero_row = row(torch.zeros((1, 1, width), device=device))
            report.compare(f"linear/{label}/{mode}/zero_column_bias", zero_col,
                           b1[left:right].to(zero_col.dtype)[None, None], exact=True)
            report.compare(f"linear/{label}/{mode}/zero_row_bias_once", zero_row,
                           b2.to(zero_row.dtype)[None, None], exact=True)


def check_loaded_model(report, reference, model, rank, rms_type) -> None:
    refs = dict(reference.named_parameters())
    mismatches = []
    for module_name, module in model.named_modules():
        for parameter_name, value in module.named_parameters(recurse=False):
            name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            expected = refs[name]
            if isinstance(module, ColumnParallelLinear):
                width = value.shape[0]
                expected = expected[rank * width:(rank + 1) * width]
            elif isinstance(module, RowParallelLinear) and parameter_name == "weight":
                width = value.shape[1]
                expected = expected[:, rank * width:(rank + 1) * width]
            elif isinstance(module, rms_type):
                width = value.shape[0]
                expected = expected[rank * width:(rank + 1) * width]
            if value.dtype != expected.dtype or not torch.equal(value, expected):
                mismatches.append(name)
    report.invariant("model/all_loaded_parameters", not mismatches, mismatches=mismatches)
    report.invariant("model/rope_frequencies", model.freqs.dtype == torch.complex128
                     and torch.equal(model.freqs, reference.freqs.to(model.freqs.device)))


@torch.no_grad()
def layer_cases(report, reference, model, device, group, rope_apply) -> None:
    dim = model.dim
    rank = dist.get_rank(group)
    tp_size = dist.get_world_size(group)
    width = dim // tp_size
    left, right = rank * width, (rank + 1) * width
    ref_block, tp_block = reference.blocks[0], model.blocks[0]
    grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
    seq_lens = torch.tensor([32], dtype=torch.long)
    padded = 40
    heads = model.num_heads
    head_dim = dim // heads
    for bf16 in (False, True):
        mode = "bf16" if bf16 else "fp32"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            # Unequal channel energy detects an incorrect local-only RMSNorm.
            q = shared_random((1, padded, dim), device, group, 301)
            q[..., :dim // 2] *= 0.125
            q[..., dim // 2:] *= 3
            if bf16:
                q = q.bfloat16()
            for attention_name in ("self_attn", "cross_attn"):
                for norm_name in ("norm_q", "norm_k"):
                    full_norm = getattr(getattr(ref_block, attention_name), norm_name)
                    tp_norm = getattr(getattr(tp_block, attention_name), norm_name)
                    report.compare(f"rmsnorm/{mode}/{attention_name}/{norm_name}",
                                   tp_norm(q[..., left:right].contiguous()), full_norm(q)[..., left:right],
                                   rtol=0.02 if bf16 else 1e-4, atol=0.02 if bf16 else 1e-5)

            rope_input = shared_random((1, padded, heads, head_dim), device, group, 302)
            if bf16:
                rope_input = rope_input.bfloat16()
            local_rope = rope_apply(rope_input[:, :, rank * (heads // tp_size):(rank + 1) * (heads // tp_size)],
                                    grid, model.freqs)
            full_rope = rope_apply(rope_input, grid, reference.freqs.to(device))
            report.compare(f"rope/{mode}/global_coordinates", local_rope,
                           full_rope[:, :, rank * (heads // tp_size):(rank + 1) * (heads // tp_size)], exact=True)

            x = shared_random((1, padded, dim), device, group, 303)
            ctx = shared_random((1, 512, dim), device, group, 304)
            ref_self = ref_block.self_attn(x, seq_lens, grid, reference.freqs.to(device))
            tp_self = tp_block.self_attn(x, seq_lens, grid, model.freqs)
            # Official FlashAttention uses half-precision kernels even when the
            # outer autocast is disabled. Do not label this a pure FP32 kernel.
            report.compare(f"self_attention/{mode}/padded", tp_self, ref_self, rtol=0.02, atol=0.02)
            altered = x.clone()
            altered[:, 32:] = shared_random((1, padded - 32, dim), device, group, 305) * 10
            changed = tp_block.self_attn(altered, seq_lens, grid, model.freqs)
            report.compare(f"self_attention/{mode}/padding_kv_excluded", changed[:, :32], tp_self[:, :32], exact=True)
            rank_agreement(report, f"self_attention/{mode}/rank_agreement", tp_self, group)

            for text_lens in (None, torch.tensor([17], dtype=torch.long)):
                tag = "all_512" if text_lens is None else "explicit_17"
                ref_cross = ref_block.cross_attn(x, ctx, text_lens)
                tp_cross = tp_block.cross_attn(x, ctx, text_lens)
                report.compare(f"cross_attention/{mode}/{tag}", tp_cross, ref_cross, rtol=0.02, atol=0.02)
                rank_agreement(report, f"cross_attention/{mode}/{tag}/rank_agreement", tp_cross, group)
            # With None, the tail remains part of text K/V; with explicit 17,
            # changing positions after 17 must not affect the output.
            changed_ctx = ctx.clone()
            changed_ctx[:, 17:] *= -3
            lens = torch.tensor([17], dtype=torch.long)
            masked = tp_block.cross_attn(x, ctx, lens)
            masked_changed = tp_block.cross_attn(x, changed_ctx, lens)
            report.compare(f"cross_attention/{mode}/explicit_length_mask", masked_changed, masked, exact=True)
            unmasked = tp_block.cross_attn(x, ctx, None)
            unmasked_changed = tp_block.cross_attn(x, changed_ctx, None)
            report.invariant(f"cross_attention/{mode}/none_includes_tail",
                             bool(torch.isfinite(unmasked_changed).all().item())
                             and not torch.equal(unmasked, unmasked_changed))

            e = shared_random((1, 6, dim), device, group, 306) * 0.1
            kwargs = dict(e=e, seq_lens=seq_lens, grid_sizes=grid,
                          context=ctx, context_lens=None)
            ref_out = ref_block(x, freqs=reference.freqs.to(device), **kwargs)
            tp_out = tp_block(x, freqs=model.freqs, **kwargs)
            report.compare(f"block0/{mode}/padded", tp_out, ref_out, rtol=0.02, atol=0.02)
            rank_agreement(report, f"block0/{mode}/rank_agreement", tp_out, group)


@torch.no_grad()
def full_model_cases(report, reference, model, device, group) -> None:
    for case, (frames, height, width, pad, text_len, timestep) in enumerate(
        ((2, 8, 8, 0, 7, 500), (3, 8, 10, 4, 33, 50))
    ):
        latent = shared_random((model.in_dim, frames, height, width), device, group, 401 + case)
        context = shared_random((text_len, model.text_dim), device, group, 411 + case)
        t = torch.tensor([timestep], device=device, dtype=torch.float32)
        seq_len = math.prod((frames // model.patch_size[0], height // model.patch_size[1],
                             width // model.patch_size[2])) + pad
        for bf16 in (False, True):
            mode = "bf16" if bf16 else "fp32"
            prefix = f"dit/case={case}/{mode}"
            ref_states, tp_states = {}, {}
            handles = []

            def capture(storage, index):
                def hook(module, inputs, output):
                    storage[index] = output.detach().clone()
                return hook

            for index, (ref_block, tp_block) in enumerate(zip(reference.blocks, model.blocks)):
                handles.append(ref_block.register_forward_hook(capture(ref_states, index)))
                handles.append(tp_block.register_forward_hook(capture(tp_states, index)))
            print(f"[rank {report.rank}] START {prefix} seq_len={seq_len} text_len={text_len}", flush=True)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    ref_out = reference([latent.clone()], t, [context.clone()], seq_len)
                    tp_out = model([latent.clone()], t, [context.clone()], seq_len)
            finally:
                for handle in handles:
                    handle.remove()
            report.invariant(prefix + "/all_blocks_executed",
                             len(ref_states) == len(model.blocks) == len(tp_states))
            for index in range(len(model.blocks)):
                report.compare(prefix + f"/block={index}", tp_states[index], ref_states[index], rtol=0.02, atol=0.02)
                rank_agreement(report, prefix + f"/block={index}/rank_agreement", tp_states[index], group)
            report.compare(prefix + "/output", tp_out[0], ref_out[0], rtol=0.02, atol=0.02)
            report.invariant(prefix + "/output_shape", list(tp_out[0].shape) == [model.out_dim, frames, height, width])
            rank_agreement(report, prefix + "/output_rank_agreement", tp_out[0], group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--wan-dir", type=Path, default=Path("../Wan2.1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    timeout = timedelta(seconds=args.timeout_seconds)
    report = None
    fixture_dir = None
    completed = False
    try:
        device = init_distributed_environment(timeout=timeout)
        initialize_model_parallel(args.tp_size, timeout=timeout)
        group = get_tensor_model_parallel_group()
        rank = dist.get_rank(group)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        commit = subprocess.check_output(["git", "-C", str(args.wan_dir), "rev-parse", "HEAD"], text=True).strip()
        if commit != "9737cba9c1c3c4d04b33fcad41c111989865d315":
            raise ValueError(f"Unexpected official Wan commit: {commit}")
        report = Report(args.output_dir / f"rank_{rank}.json", rank, dict(
            torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device), device=str(device),
            wan_commit=commit, checkpoint=str(args.ckpt_dir.resolve()), tp_size=dist.get_world_size(group),
            parameter_dtype="float32", tf32=False,
            precision_policy="Original full-reference tolerances retained; unmet approximate criteria are REVIEW, not accepted.",
            scope="Linear/FFN, RMSNorm, RoPE, Attention, block0, four small full-DiT forwards; no T5/VAE/denoising/performance.",
        ))
        # Bind CUDA first: importing official Wan can query the active device.
        from models.wan.tp_wan import create_wan_tp_model, load_wan_tp_weights
        from parallel.tp_attention import TPWanRMSNorm
        from wan.modules.model import WanModel, rope_apply
        from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
        import wan.modules.model as official_model

        imported_root = Path(official_model.__file__).resolve().parents[2]
        if imported_root != args.wan_dir.resolve():
            raise ValueError(f"Imported Wan from {imported_root}, expected {args.wan_dir.resolve()}")
        report.data["metadata"]["imported_wan_root"] = str(imported_root)
        report.data["metadata"]["attention_backend"] = "FA3" if FLASH_ATTN_3_AVAILABLE else "FA2"
        if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
            raise RuntimeError("Official Wan flash_attention requires FlashAttention")
        paths = [None]
        if rank == 0:
            fixture_dir = TemporaryDirectory(prefix="wan_tp_suite_")
            path = Path(fixture_dir.name) / "linear.safetensors"
            generator = torch.Generator().manual_seed(20261011)
            save_file({
                "blocks.0.ffn.0.weight": torch.randn(8960, 1536, generator=generator) / math.sqrt(1536),
                "blocks.0.ffn.0.bias": torch.linspace(0.25, 0.75, 8960),
                "blocks.0.ffn.2.weight": torch.randn(1536, 8960, generator=generator) / math.sqrt(8960),
                "blocks.0.ffn.2.bias": torch.linspace(-0.75, -0.25, 1536),
            }, str(path))
            paths[0] = str(path)
        dist.broadcast_object_list(paths, src=0, group=group, device=device)
        linear_cases(report, Path(paths[0]), "synthetic", device, group)
        linear_cases(report, args.ckpt_dir / "diffusion_pytorch_model.safetensors", "checkpoint", device, group)
        # Linear temporaries have left scope; do not hold two test allocations.
        torch.cuda.empty_cache()
        print(f"[rank {rank}] Loading official and TP DiT once (no T5/VAE)", flush=True)
        reference = WanModel.from_pretrained(str(args.ckpt_dir), torch_dtype=torch.float32)
        reference = reference.eval().requires_grad_(False).to(device)
        reference.freqs = reference.freqs.to(device)
        with (args.ckpt_dir / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        model = create_wan_tp_model(config, group, device=device, dtype=torch.float32)
        load_wan_tp_weights(model, args.ckpt_dir / "diffusion_pytorch_model.safetensors")
        model.eval().requires_grad_(False)
        check_loaded_model(report, reference, model, rank, TPWanRMSNorm)
        layer_cases(report, reference, model, device, group, rope_apply)
        full_model_cases(report, reference, model, device, group)
        checks = report.data["checks"]
        counts = {status: sum(c["status"] == status for c in checks) for status in ("PASS", "REVIEW", "FAIL")}
        stage_counts = {}
        for check in checks:
            stage = check["name"].split("/")[0]
            stage_counts.setdefault(stage, dict(PASS=0, REVIEW=0, FAIL=0))[check["status"]] += 1
        severity = 2 if counts["FAIL"] else (1 if counts["REVIEW"] else 0)
        overall = torch.tensor(severity, device=device)
        dist.all_reduce(overall, op=dist.ReduceOp.MAX, group=group)
        report.data["summary"] = dict(status=("PASS", "REVIEW", "FAIL")[overall.item()], local_counts=counts,
                                      stages=stage_counts,
                                      numerical_failures_do_not_abort=True,
                                      exit_zero_means="suite completed, inspect summary.status")
        report.save()
        dist.barrier(group=group)
        completed = True
    finally:
        try:
            try:
                destroy_model_parallel()
            finally:
                destroy_distributed_environment()
        finally:
            if fixture_dir is not None:
                fixture_dir.cleanup()
            if report is not None:
                report.save()
    # Reached only after cleanup succeeds. Execution exceptions keep complete=false.
    if completed:
        report.data["complete"] = True
        report.save()
        print(f"[rank {report.rank}] COMPLETED: {report.data['summary']}; report={report.path}", flush=True)


if __name__ == "__main__":
    main()
