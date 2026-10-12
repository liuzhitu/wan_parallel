"""Check the production WORLD/TP lifecycle and NCCL without loading Wan."""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from runtime.distributed import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from runtime.parallel_state import (
    destroy_model_parallel,
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    timeout = timedelta(seconds=args.timeout_seconds)

    def report(message: str) -> None:
        print(f"[rank {os.environ['RANK']}] {message}", flush=True)

    try:
        report("Initializing WORLD and binding GPU")
        device = init_distributed_environment(timeout=timeout)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        report(
            f"WORLD ready: size={world_size}, backend={dist.get_backend()}, "
            f"LOCAL_RANK={os.environ['LOCAL_RANK']}, device={device}, "
            f"current_device={torch.cuda.current_device()}, "
            f"GPU={torch.cuda.get_device_name(device)}"
        )

        report("Checking WORLD AllReduce")
        value = torch.tensor([rank + 1], device=device, dtype=torch.float32)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) // 2
        torch.testing.assert_close(
            value, torch.full_like(value, expected_sum), rtol=0, atol=0
        )
        report(f"WORLD AllReduce PASS: sum={value.item()}")

        report("Initializing TP group")
        initialize_model_parallel(args.tp_size, timeout=timeout)
        group = get_tensor_model_parallel_group()
        tp_rank = dist.get_rank(group)
        tp_size = dist.get_world_size(group)
        report(f"TP ready: rank={tp_rank}, size={tp_size}, backend={dist.get_backend(group)}")

        for dtype in (torch.float32, torch.bfloat16):
            for numel in (4096, 1048576):
                report(f"Checking TP AllReduce and broadcast: dtype={dtype}, numel={numel}")
                value = torch.full((numel,), tp_rank + 1, device=device, dtype=dtype)
                dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
                expected = torch.full_like(value, tp_size * (tp_size + 1) // 2)
                torch.testing.assert_close(value, expected, rtol=0, atol=0)

                # The current pure TP group contains every WORLD rank, so src=0
                # is both the global source rank and TP rank zero.
                value.fill_(7 if tp_rank == 0 else -1)
                dist.broadcast(value, src=0, group=group)
                torch.testing.assert_close(value, torch.full_like(value, 7), rtol=0, atol=0)
                report(f"TP AllReduce + broadcast PASS: dtype={dtype}, numel={numel}")

        report("Checking TP barrier")
        dist.barrier(group=group)
        torch.cuda.synchronize(device)
        report("Communication checks PASS; destroying TP then WORLD")
    finally:
        # Match the generation entry point; do not add a barrier on error paths.
        try:
            destroy_model_parallel()
        finally:
            destroy_distributed_environment()

    report("PASS: communication checks and process-group cleanup completed")


if __name__ == "__main__":
    main()
