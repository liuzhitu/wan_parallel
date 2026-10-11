"""Own the tensor-parallel group separately from the default WORLD group.

This project currently supports one pure TP inference group. The run entry point
initializes WORLD first, configures TP size, and destroys TP before WORLD. Models
receive the TP ProcessGroup explicitly; ranks and sizes are queried from it.
"""

from __future__ import annotations

from datetime import timedelta

import torch.distributed as dist


_TP_GROUP: dist.ProcessGroup | None = None


def initialize_model_parallel(
    tensor_model_parallel_size: int,
    *,
    timeout: timedelta,
) -> None:
    """Create one NCCL TP group; all WORLD ranks must call this together.

    WORLD size must equal TP size because DP and PP are not implemented. Even
    with identical members, TP is a separate group with its own lifetime.
    """
    global _TP_GROUP
    if not dist.is_initialized():
        raise RuntimeError("Initialize the distributed environment before model parallelism")
    if _TP_GROUP is not None:
        raise RuntimeError("The tensor-parallel group is already initialized")

    world_size = dist.get_world_size()
    if world_size != tensor_model_parallel_size:
        raise ValueError("Pure TP requires WORLD size equal to tensor_model_parallel_size")

    _TP_GROUP = dist.new_group(
        ranks=list(range(world_size)),
        backend="nccl",
        timeout=timeout,
    )


def get_tensor_model_parallel_group() -> dist.ProcessGroup:
    """Return the initialized group used by TP layers and checkpoint loaders."""
    if _TP_GROUP is None:
        raise RuntimeError("The tensor-parallel group is not initialized")
    return _TP_GROUP


def destroy_model_parallel() -> None:
    """Destroy TP before WORLD; a group not yet created needs no cleanup."""
    global _TP_GROUP
    if _TP_GROUP is not None:
        dist.destroy_process_group(_TP_GROUP)
        _TP_GROUP = None
