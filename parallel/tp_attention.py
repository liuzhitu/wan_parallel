"""Inference tensor parallelism for fixed-version Wan2.1 T2V attention.

Each rank receives the same full sequence and owns a contiguous range of heads.
The caller owns process-group initialization and must load all parameters before
forward. Constructors allocate uninitialized shards, including on meta; weight
loading and model replacement are separate from this module. All ranks must enter
the Q/K statistics and output collectives in the same order. Training is outside
this module's contract.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from wan.modules.attention import flash_attention
from wan.modules.model import rope_apply

from .tp_linear import ColumnParallelLinear, RowParallelLinear


class TPWanRMSNorm(nn.Module):
    """Normalize a feature shard using statistics over the full hidden dimension.

    The affine weight owns the same contiguous channels as its Q or K projection.
    Statistics are FP32; normalization casts back before multiplying the affine
    weight, matching WanRMSNorm's dtype order.
    """

    def __init__(
        self,
        dim: int,
        process_group: dist.ProcessGroup | None,
        device: torch.device | str,
        dtype: torch.dtype,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size(process_group)
        if tp_size < 1 or dim % tp_size:
            raise ValueError("TPWanRMSNorm requires group membership and dim divisible by TP size")
        self.dim = dim
        self.eps = eps
        self.process_group = process_group
        self.weight = nn.Parameter(
            torch.empty(dim // tp_size, device=device, dtype=dtype),
            requires_grad=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        square_sum = x_float.pow(2).sum(dim=-1, keepdim=True)
        dist.all_reduce(square_sum, op=dist.ReduceOp.SUM, group=self.process_group)
        normalized = x_float * torch.rsqrt(square_sum / self.dim + self.eps)
        return normalized.type_as(x) * self.weight


class TPWanSelfAttention(nn.Module):
    """Column Q/K/V, full-dimension Q/K RMSNorm, local heads, and Row output.

    Input and output are [B, L, dim] on every rank. Only the head dimension is
    partitioned, so the original video grid, RoPE frequencies and K/V lengths
    remain valid without rank-dependent sequence offsets.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        process_group: dist.ProcessGroup | None,
        device: torch.device | str,
        dtype: torch.dtype,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size(process_group)
        if dim % num_heads:
            raise ValueError("TPWanSelfAttention requires dim divisible by num_heads")
        if tp_size < 1 or num_heads % tp_size:
            raise ValueError("TPWanSelfAttention requires group membership and whole heads per rank")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = ColumnParallelLinear(dim, dim, process_group, device, dtype)
        self.k = ColumnParallelLinear(dim, dim, process_group, device, dtype)
        self.v = ColumnParallelLinear(dim, dim, process_group, device, dtype)
        self.o = RowParallelLinear(dim, dim, process_group, device, dtype)
        self.norm_q = (
            TPWanRMSNorm(dim, process_group, device, dtype, eps)
            if qk_norm else nn.Identity()
        )
        self.norm_k = (
            TPWanRMSNorm(dim, process_group, device, dtype, eps)
            if qk_norm else nn.Identity()
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        b, s = x.shape[:2]
        n = self.q.weight.shape[0] // self.head_dim
        d = self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        output = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        return self.o(output.flatten(2))


class TPWanT2VCrossAttention(TPWanSelfAttention):
    """Video Q attends to replicated text context using rank-local heads.

    As in WanT2VCrossAttention, no RoPE is applied and context_lens is passed
    through. WanModel supplies None to include all padded text positions.
    """

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        b = x.size(0)
        n = self.q.weight.shape[0] // self.head_dim
        d = self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        output = flash_attention(q, k, v, k_lens=context_lens)
        return self.o(output.flatten(2))
