"""Construct Wan2.1 T2V with TP attention/FFN and load all DiT parameters."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from parallel.tp_attention import TPWanSelfAttention, TPWanT2VCrossAttention
from parallel.tp_linear import ColumnParallelLinear, RowParallelLinear
from parallel.tp_weight_loader import (
    load_attention_weights,
    load_linear_weight,
    load_replicated_weights,
)
from wan.modules.model import WanModel, rope_params


def create_wan_tp_model(
    config: Mapping[str, Any],
    process_group: dist.ProcessGroup | None,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> WanModel:
    """Allocate a T2V WanModel with TP attention/FFN, without loading weights.

    The caller supplies the official model config and an initialized process
    group. Construct and replace on meta before allocating the final parameter
    shapes. FP32 parameters match the existing BF16-autocast baseline.
    All checkpoint parameters must be loaded before the first forward.
    """
    with torch.device("meta"):
        model = WanModel.from_config(dict(config))
        model.to(dtype=dtype)
        parallelize_wan_model(
            model,
            process_group,
            device="meta",
            dtype=dtype,
        )

    model.to_empty(device=device)

    # Wan keeps freqs as a plain tensor attribute, so to_empty does not
    # materialize it. Reuse the official CPU formula and preserve complex128.
    head_dim = model.dim // model.num_heads
    with torch.device("cpu"):
        model.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        ).to(device=model.patch_embedding.weight.device)

    return model


def parallelize_wan_model(
    model: WanModel,
    process_group: dist.ProcessGroup | None,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> WanModel:
    """Replace T2V FFN and self/cross attention before checkpoint loading.

    Call once with an initialized process group. For meta construction, pass
    device="meta" and materialize the final model with to_empty before loading.
    No parameter data is copied; the original modules supply dimensions and
    attention settings. GELU, block forward, modulation, residuals and the
    model's full sequence stay unchanged.
    """
    if model.model_type != "t2v":
        raise ValueError("Wan TP currently supports only T2V models")

    for block in model.blocks:
        fc1, activation, fc2 = block.ffn
        block.ffn = nn.Sequential(
            ColumnParallelLinear(
                in_features=fc1.in_features,
                out_features=fc1.out_features,
                process_group=process_group,
                device=device,
                dtype=dtype,
            ),
            activation,
            RowParallelLinear(
                in_features=fc2.in_features,
                out_features=fc2.out_features,
                process_group=process_group,
                device=device,
                dtype=dtype,
            ),
        )

        for name, attention_type in (
            ("self_attn", TPWanSelfAttention),
            ("cross_attn", TPWanT2VCrossAttention),
        ):
            attention = getattr(block, name)
            setattr(
                block,
                name,
                attention_type(
                    dim=attention.dim,
                    num_heads=attention.num_heads,
                    process_group=process_group,
                    device=device,
                    dtype=dtype,
                    window_size=attention.window_size,
                    qk_norm=attention.qk_norm,
                    eps=attention.eps,
                ),
            )

    return model


def load_wan_tp_weights(
    model: WanModel,
    checkpoint_path: str | PathLike[str],
) -> None:
    """Load all T2V DiT parameters, combining TP shards and replicated weights.

    Call after parallelize_wan_model and storage allocation. Attention loading
    includes Q/K RMSNorm affine shards when qk_norm is enabled. Uses the original
    Wan checkpoint names. Remaining parameters are loaded in full on every rank.
    RoPE freqs are rebuilt by create_wan_tp_model rather than checkpoint loading.
    All ranks must use the same checkpoint.
    """

    for i, block in enumerate(model.blocks):
        load_linear_weight(
            block.ffn[0],
            checkpoint_path,
            prefix=f"blocks.{i}.ffn.0",
        )

        load_linear_weight(
            block.ffn[2],
            checkpoint_path,
            prefix=f"blocks.{i}.ffn.2",
        )
        for name in ("self_attn", "cross_attn"):
            load_attention_weights(
                getattr(block, name),
                checkpoint_path,
                prefix=f"blocks.{i}.{name}",
            )

    load_replicated_weights(model, checkpoint_path)
