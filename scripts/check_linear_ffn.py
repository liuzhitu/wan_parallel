"""Compare TP Column, Row and Wan FFN against full Linear references."""

from __future__ import annotations

import argparse
import math
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
    destroy_model_parallel,
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)


def make_fixture(path: Path) -> None:
    """Create a checkpoint fixture, never a separate TP initialization path."""
    generator = torch.Generator().manual_seed(20261011)
    save_file({
        "blocks.0.ffn.0.weight": torch.randn(8960, 1536, generator=generator) / math.sqrt(1536),
        "blocks.0.ffn.0.bias": torch.linspace(0.25, 0.75, 8960),
        "blocks.0.ffn.2.weight": torch.randn(1536, 8960, generator=generator) / math.sqrt(8960),
        "blocks.0.ffn.2.bias": torch.linspace(-0.75, -0.25, 1536),
    }, str(path))


@torch.no_grad()
def check_checkpoint(path: Path, label: str, device: torch.device, group: dist.ProcessGroup) -> bool:
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        w1, b1, w2, b2 = [checkpoint.get_tensor(f"blocks.0.ffn.{name}").to(device=device, dtype=torch.float32)
                          for name in ("0.weight", "0.bias", "2.weight", "2.bias")]
    hidden, dim = w1.shape
    column = ColumnParallelLinear(dim, hidden, group, device, torch.float32)
    row = RowParallelLinear(hidden, dim, group, device, torch.float32)
    parameter_ids = tuple(id(p) for layer in (column, row) for p in layer.parameters())
    load_linear_weight(column, path, "blocks.0.ffn.0")
    load_linear_weight(row, path, "blocks.0.ffn.2")
    width = hidden // size
    start, end = rank * width, (rank + 1) * width
    checks: list[bool] = []

    def same_order_reference(x: torch.Tensor, *, ffn: bool) -> torch.Tensor:
        """Reproduce TP GEMM sizes/rounding on one GPU, without collectives.

        This entry point requires TP2. Adding the two local outputs reproduces
        the two-rank SUM rounding; the full bias is added only afterwards.
        Reference tensors come from the checkpoint, never from TP outputs.
        """
        partials = []
        for shard in range(size):
            left, right = shard * width, (shard + 1) * width
            if ffn:
                intermediate = F.gelu(
                    F.linear(x, w1[left:right].contiguous(), b1[left:right].contiguous()),
                    approximate="tanh",
                )
            else:
                intermediate = x[..., left:right].contiguous()
            partials.append(F.linear(intermediate, w2[:, left:right].contiguous(), bias=None))
        result = partials[0] + partials[1]
        return result.add_(b2)

    def compare(name: str, actual: torch.Tensor, expected: torch.Tensor,
                *, rtol: float, atol: float) -> None:
        # Report errors in FP32 even for BF16 outputs; do not divide elementwise
        # by near-zero reference values.
        delta = actual.float() - expected.float()
        max_abs = delta.abs().max().item()
        ref_norm = torch.linalg.vector_norm(expected.float()).item()
        delta_norm = torch.linalg.vector_norm(delta).item()
        relative_l2 = delta_norm / ref_norm if ref_norm else (0.0 if delta_norm == 0 else math.inf)
        passed = actual.dtype == expected.dtype and bool(torch.isfinite(actual).all().item()) and bool(
            torch.allclose(actual, expected, rtol=rtol, atol=atol)
        )
        checks.append(passed)
        bad_elements = int((~torch.isclose(actual, expected, rtol=rtol, atol=atol)).sum().item())
        print(f"[rank {rank}] {label}/{name}: {'PASS' if passed else 'FAIL'} "
              f"max_abs={max_abs:.6g} relative_l2={relative_l2:.6g} "
              f"bad_elements={bad_elements}/{actual.numel()} "
              f"dtype={actual.dtype} expected_dtype={expected.dtype} "
              f"rtol={rtol:g} atol={atol:g}", flush=True)

    # Loader axes, bias ownership and in-place filling are checked independently
    # of floating-point GEMM error.
    compare("load/column.weight", column.weight, w1[start:end], rtol=0, atol=0)
    compare("load/column.bias", column.bias, b1[start:end], rtol=0, atol=0)
    compare("load/row.weight", row.weight, w2[:, start:end], rtol=0, atol=0)
    compare("load/row.bias", row.bias, b2, rtol=0, atol=0)
    same_parameters = parameter_ids == tuple(id(p) for layer in (column, row) for p in layer.parameters())
    checks.append(same_parameters)
    print(f"[rank {rank}] {label}/load/parameter_identity: "
          f"{'PASS' if same_parameters else 'FAIL'}", flush=True)

    generator = torch.Generator(device=device).manual_seed(42)
    for autocast in (False, True):
        mode = "bf16_autocast" if autocast else "fp32"
        # Initial screening thresholds: FP32 permits reordered accumulation;
        # BF16 permits rounding of local partial sums and the NCCL reduction.
        # Print raw errors as well; these thresholds are not model-level bounds.
        rtol, atol = (2e-2, 2e-2) if autocast else (1e-4, 1e-5)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            for tokens in (1, 17, 128):
                x = torch.randn((1, tokens, dim), generator=generator, device=device)
                dist.broadcast(x, src=0, group=group)
                reference_column = F.linear(x, w1, b1)
                local = column(x)
                compare(f"{mode}/tokens={tokens}/column", local,
                        reference_column[..., start:end], rtol=rtol, atol=atol)

                row_input = torch.randn((1, tokens, hidden), generator=generator, device=device)
                dist.broadcast(row_input, src=0, group=group)
                actual_row = row(row_input[..., start:end].contiguous())
                compare(f"{mode}/tokens={tokens}/row",
                        actual_row,
                        F.linear(row_input, w2, b2), rtol=rtol, atol=atol)

                actual_ffn = row(F.gelu(local, approximate="tanh"))
                compare(f"{mode}/tokens={tokens}/ffn",
                        actual_ffn,
                        F.linear(F.gelu(reference_column, approximate="tanh"), w2, b2),
                        rtol=rtol, atol=atol)

                if autocast:
                    # Keep the original full-GEMM acceptance checks above. These
                    # extra exact checks diagnose whether a failure originates
                    # in distributed execution or in a different rounding path.
                    compare(f"{mode}/tokens={tokens}/row_same_order",
                            actual_row, same_order_reference(row_input, ffn=False), rtol=0, atol=0)
                    compare(f"{mode}/tokens={tokens}/ffn_same_order",
                            actual_ffn, same_order_reference(x, ffn=True), rtol=0, atol=0)

            # Pure bias checks remove all GEMM accumulation error. For synthetic
            # weights a duplicate Row bias would differ by at least 0.25.
            zero_column = column(torch.zeros((1, 1, dim), device=device))
            compare(f"{mode}/zero_input/column_bias", zero_column,
                    b1[start:end].to(zero_column.dtype)[None, None], rtol=0, atol=0)
            zero_row = row(torch.zeros((1, 1, width), device=device))
            compare(f"{mode}/zero_input/row_bias_once", zero_row,
                    b2.to(zero_row.dtype)[None, None], rtol=0, atol=0)
    return all(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    timeout = timedelta(seconds=args.timeout_seconds)
    fixture_dir = None
    passed = False
    try:
        device = init_distributed_environment(timeout=timeout)
        initialize_model_parallel(2, timeout=timeout)
        group = get_tensor_model_parallel_group()
        rank = dist.get_rank(group)
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"[rank {rank}] device={device}; FP32 parameters; TF32 disabled; "
              "testing synthetic and checkpoint block 0 FFN", flush=True)

        # Single-node torchrun ranks share /tmp. Only rank zero writes the fixture.
        paths = [None]
        if rank == 0:
            fixture_dir = TemporaryDirectory(prefix="wan_tp_linear_")
            path = Path(fixture_dir.name) / "fixture.safetensors"
            make_fixture(path)
            paths[0] = str(path)
        dist.broadcast_object_list(paths, src=0, group=group, device=device)
        synthetic_ok = check_checkpoint(Path(paths[0]), "synthetic", device, group)
        checkpoint_ok = check_checkpoint(
            args.ckpt_dir / "diffusion_pytorch_model.safetensors", "checkpoint", device, group
        )
        # Numerical failures do not interrupt another rank's later collectives.
        # Finish every case, then make the final status agree across ranks.
        status = torch.tensor(int(synthetic_ok and checkpoint_ok), device=device)
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
        passed = bool(status.item())
        dist.barrier(group=group)
    finally:
        try:
            destroy_model_parallel()
        finally:
            destroy_distributed_environment()
        if fixture_dir is not None:
            fixture_dir.cleanup()

    if not passed:
        raise SystemExit(
            "FAIL: at least one Linear/FFN check failed; compare full-reference "
            "and same_order results before changing tolerances or implementation"
        )
    print("PASS: Column, Row and FFN checks completed on all ranks; groups cleaned up", flush=True)


if __name__ == "__main__":
    main()
