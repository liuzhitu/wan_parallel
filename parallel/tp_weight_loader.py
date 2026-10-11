"""Load Wan checkpoint slices into already allocated TP parameters.

The layer's process group determines TP size and group-local rank; callers do
not pass a separate rank. Parameter device and dtype are owned by the layer.
Loading changes their contents in place, without replacing Parameter objects.
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn

from .tp_linear import ColumnParallelLinear, RowParallelLinear

if TYPE_CHECKING:
    from .tp_attention import TPWanRMSNorm, TPWanSelfAttention


@torch.no_grad()
def load_linear_weight(
    layer: ColumnParallelLinear | RowParallelLinear,
    checkpoint_path: str | PathLike[str],
    prefix: str,
) -> None:
    """Load the rank-local weight slice and bias into an existing TP layer."""
    from safetensors import safe_open

    shard_dim = layer.shard_dim
    tp_size = dist.get_world_size(layer.process_group)
    tp_rank = dist.get_rank(layer.process_group)
    expected_shape = list(layer.weight.shape)
    expected_shape[shard_dim] *= tp_size
    width = layer.weight.shape[shard_dim]
    start = tp_rank * width
    end = start + width

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        weight_view = checkpoint.get_slice(f"{prefix}.weight")
        bias_view = checkpoint.get_slice(f"{prefix}.bias")
        # Check the full shapes before slicing: copy_ permits broadcasting,
        # which could otherwise silently populate a layer with wrong weights.
        if weight_view.get_shape() != expected_shape:
            raise ValueError(f"{prefix}.weight: expected checkpoint shape {expected_shape}")
        if bias_view.get_shape() != [expected_shape[0]]:
            raise ValueError(f"{prefix}.bias: expected checkpoint shape {[expected_shape[0]]}")
        if shard_dim == 0:
            weight = weight_view[start:end, :]
            bias = bias_view[start:end]
        else:
            weight = weight_view[:, start:end]
            bias = checkpoint.get_tensor(f"{prefix}.bias")

        # Prepare contiguous CPU slices in the destination dtype, then copy
        # directly into existing storage. No explicit GPU staging tensor.
        layer.weight.copy_(weight.to(dtype=layer.weight.dtype).contiguous())
        layer.bias.copy_(bias.to(dtype=layer.bias.dtype).contiguous())


@torch.no_grad()
def load_rmsnorm_weight(
    layer: TPWanRMSNorm,
    checkpoint_path: str | PathLike[str],
    prefix: str,
) -> None:
    """Load the affine shard matching the rank-local Q or K channels.

    The checkpoint stores the full [dim] weight and no bias. Call after the
    destination parameter storage has been allocated, as for Linear loading.
    """
    from safetensors import safe_open

    tp_rank = dist.get_rank(layer.process_group)
    width = layer.weight.shape[0]
    start = tp_rank * width
    end = start + width

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        weight_view = checkpoint.get_slice(f"{prefix}.weight")
        # Validate before slicing so an oversized checkpoint cannot silently
        # supply a plausible local shard, and copy_ cannot broadcast a wrong one.
        if weight_view.get_shape() != [layer.dim]:
            raise ValueError(f"{prefix}.weight: expected checkpoint shape {[layer.dim]}")
        weight = weight_view[start:end]
        layer.weight.copy_(weight.to(dtype=layer.weight.dtype).contiguous())


def load_attention_weights(
    layer: TPWanSelfAttention,
    checkpoint_path: str | PathLike[str],
    prefix: str,
) -> None:
    """Load one TP self-attention or T2V cross-attention module in place.

    For example, prefix="blocks.0.self_attn" uses the original Wan checkpoint
    names. The T2V cross-attention subclass has the same parameter layout.
    """
    for name in ("q", "k", "v", "o"):
        load_linear_weight(
            getattr(layer, name), checkpoint_path, prefix=f"{prefix}.{name}"
        )

    if layer.qk_norm:
        load_rmsnorm_weight(
            layer.norm_q, checkpoint_path, prefix=f"{prefix}.norm_q"
        )
        load_rmsnorm_weight(
            layer.norm_k, checkpoint_path, prefix=f"{prefix}.norm_k"
        )


@torch.no_grad()
def load_replicated_weights(
    model: nn.Module,
    checkpoint_path: str | PathLike[str],
) -> None:
    """Load full parameters outside TP Linear/RMSNorm into existing storage.

    Module paths are the original Wan checkpoint keys. TP parameter holders are
    loaded separately, including RowParallelLinear's replicated bias. The fixed
    T2V model has no checkpoint buffers; its plain RoPE freqs attribute is rebuilt
    by the model constructor. Call after destination storage allocation.
    """
    from safetensors import safe_open

    from .tp_attention import TPWanRMSNorm

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        for module_name, module in model.named_modules():
            if isinstance(module, (ColumnParallelLinear, RowParallelLinear, TPWanRMSNorm)):
                continue
            # Inspect only parameters owned by this module. Recursing here would
            # visit child TP parameters and load the same parameters repeatedly.
            for parameter_name, parameter in module.named_parameters(recurse=False):
                key = f"{module_name}.{parameter_name}" if module_name else parameter_name
                expected_shape = list(parameter.shape)
                if checkpoint.get_slice(key).get_shape() != expected_shape:
                    raise ValueError(f"{key}: expected checkpoint shape {expected_shape}")
                weight = checkpoint.get_tensor(key)
                parameter.copy_(weight.to(dtype=parameter.dtype).contiguous())
