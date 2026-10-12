"""Project-owned inference TP Linear layers for Wan2.1's biased projections.

The caller owns process-group initialization and lifetime; passing None uses
the default process group. Constructors allocate uninitialized local parameters
on the requested device and dtype. Load them with parallel.tp_weight_loader before the
first forward. All ranks must load the same checkpoint, receive matching inputs,
and enter RowParallelLinear collectives in the same order. Distributed backward
and training are outside this module's contract.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class ColumnParallelLinear(nn.Module):
    """Split weight rows and bias, producing a rank-local output feature shard.

    For TP size P, each rank owns weight [O/P, I] and bias [O/P].
    All ranks receive the same input [..., I]; each returns [..., O/P].
    No output gather is performed, so the result can feed a RowParallelLinear
    directly after a local elementwise activation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: dist.ProcessGroup | None,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.shard_dim = 0
        tp_size = dist.get_world_size(process_group)
        if tp_size < 1 or out_features % tp_size:
            raise ValueError("ColumnParallelLinear requires group membership and out_features divisible by TP size")
        self.process_group = process_group
        self.weight = nn.Parameter(
            torch.empty(
                out_features // tp_size, in_features, device=device, dtype=dtype
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(out_features // tp_size, device=device, dtype=dtype),
            requires_grad=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Sum partial outputs, including a single rank's fused bias contribution.

    For TP size P, each rank owns weight [O, I/P] and full bias [O].
    Each rank receives its contiguous feature range of the input [..., I/P].
    TP rank 0 includes the full bias in its Linear; other ranks omit it.
    SUM AllReduce restores [..., O] with exactly one bias contribution.
    The input is not split again; passing a full input is a shape error.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: dist.ProcessGroup | None,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.shard_dim = 1
        tp_size = dist.get_world_size(process_group)
        if tp_size < 1 or in_features % tp_size:
            raise ValueError("RowParallelLinear requires group membership and in_features divisible by TP size")
        self.process_group = process_group
        self.weight = nn.Parameter(
            torch.empty(
                out_features, in_features // tp_size, device=device, dtype=dtype
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype),
            requires_grad=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match vLLM's Row Linear: include bias on one TP rank before reduction.
        bias = self.bias if dist.get_rank(self.process_group) == 0 else None
        output = F.linear(x, self.weight, bias=bias)
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        return output
