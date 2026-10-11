"""Torchrun entry point for the project-owned Wan2.1 T2V TP pipeline."""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Wan2.1-T2V-1.3B video with pure tensor parallelism."
    )
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--size", default="832*480", help="Width*height; official T2V-1.3B sizes.")
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=8.0)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42, help="Negative selects a random seed on rank zero.")
    parser.add_argument(
        "--prompt",
        default=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage."
        ),
        help="Defaults to the official T2V-1.3B example prompt.",
    )
    parser.add_argument("--negative-prompt", default="", help="Empty uses the official default.")
    parser.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-file", type=Path, default=Path("outputs/wan_tp.mp4"))
    parser.add_argument(
        "--timeout-seconds", type=float, default=1800,
        help="WORLD/TP communication timeout, including waits for loading and decoding.",
    )
    args = parser.parse_args()
    if args.sample_steps <= 0:
        parser.error("--sample-steps must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main() -> None:
    args = _parse_args()
    timeout = timedelta(seconds=args.timeout_seconds)
    try:
        device = init_distributed_environment(timeout=timeout)
        initialize_model_parallel(args.tp_size, timeout=timeout)
        group = get_tensor_model_parallel_group()
        rank = dist.get_rank(group)
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s [TP rank {rank}] %(levelname)s: %(message)s",
        )

        # Official Wan imports query the current CUDA device. Bind it first.
        from pipelines.wan_t2v_tp import WanT2VTP
        from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
        from wan.utils.utils import cache_video

        config = WAN_CONFIGS["t2v-1.3B"]
        if args.size not in SUPPORTED_SIZES["t2v-1.3B"]:
            raise ValueError(f"Unsupported T2V-1.3B size: {args.size}")
        if args.frame_num < 1 or (args.frame_num - 1) % config.vae_stride[0] != 0:
            raise ValueError("--frame-num must be positive and have the form 4n+1")

        logging.info("Generation settings: %s", args)
        pipeline = WanT2VTP(
            config,
            args.ckpt_dir,
            process_group=group,
            device=device,
            t5_cpu=args.t5_cpu,
        )
        video = pipeline.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale,
            n_prompt=args.negative_prompt,
            seed=args.seed,
            offload_model=args.offload_model,
        )
        if rank == 0:
            args.save_file.parent.mkdir(parents=True, exist_ok=True)
            logging.info("Saving video to %s", args.save_file)
            saved_file = cache_video(
                tensor=video[None],
                save_file=str(args.save_file),
                fps=config.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
                retry=1,
            )
            # The official writer returns None instead of raising on failure.
            if saved_file is None:
                raise RuntimeError(f"Video save failed: {args.save_file}")

        # generate's barrier covers decoding; this boundary covers file output.
        dist.barrier(group=group)
        logging.info("Generation and video save completed.")
    finally:
        # No barrier on error paths; clean up partial initialization as well.
        try:
            destroy_model_parallel()
        finally:
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
