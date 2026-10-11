"""Initialize and destroy the default NCCL environment for a torchrun job.

The run entry point owns WORLD and must destroy model-parallel groups before
destroying this environment. GPU selection uses LOCAL_RANK; rank, world size
and rendezvous are supplied by torchrun through env://.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed_environment(*, timeout: timedelta) -> torch.device:
    """Bind the local GPU and initialize a fresh WORLD group on every rank.

    Model-parallel size is configured separately. timeout bounds rendezvous and
    collective waits; it does not provide retries or cross-rank recovery.
    """
    if dist.is_initialized():
        raise RuntimeError("The default process group is already initialized")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
        timeout=timeout,
    )
    return device


def destroy_distributed_environment() -> None:
    """Destroy WORLD after model-parallel groups, including partial-init cleanup."""
    if dist.is_initialized():
        dist.destroy_process_group()
