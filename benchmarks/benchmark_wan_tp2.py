"""Measure pure TP2 inference on two GPUs without changing the sampling path."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist

from benchmarks.benchmark_wan_single_gpu import (
    PROMPT,
    SOURCE_COMMIT,
    DiTForwardTiming,
    command_output,
    summary,
)
from runtime.distributed import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from runtime.parallel_state import (
    destroy_model_parallel,
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-dir", type=Path, default=Path("../Wan2.1"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("../models/Wan2.1-T2V-1.3B"))
    parser.add_argument("--tp-size", type=int, choices=(2,), default=2)
    parser.add_argument("--size", choices=("832*480", "480*832"), default="832*480")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, help="Optional matching single-GPU metrics.json.")
    args = parser.parse_args()
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error("--frames must be positive and have the form 4n+1")
    if min(args.steps, args.warmup, args.runs) < 1 or args.seed < 0:
        parser.error("steps, warmup and runs must be positive; seed must be nonnegative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


class TPAllReduceVolume:
    """Count completed module calls and their logical collective payloads.

    Each Row Linear and TP RMSNorm forward executes exactly one SUM AllReduce.
    Hooks inspect shapes/dtypes only, never synchronize or replace collectives.
    Bytes are per-rank tensor bytes, not NCCL bus traffic or wire bandwidth.
    """

    def __init__(self, model):
        from parallel.tp_attention import TPWanRMSNorm
        from parallel.tp_linear import RowParallelLinear

        self.handles = []
        for module in model.modules():
            if isinstance(module, RowParallelLinear):
                self.handles.append(module.register_forward_hook(self.record_row))
            elif isinstance(module, TPWanRMSNorm):
                self.handles.append(module.register_forward_hook(self.record_norm))
        self.reset()

    def reset(self):
        self.counts = {
            "row_output": {"calls": 0, "tensor_bytes": 0},
            "qk_statistics": {"calls": 0, "tensor_bytes": 0},
        }

    def record_row(self, module, inputs, output):
        counter = self.counts["row_output"]
        counter["calls"] += 1
        counter["tensor_bytes"] += output.numel() * output.element_size()

    def record_norm(self, module, inputs, output):
        counter = self.counts["qk_statistics"]
        counter["calls"] += 1
        # The actual reduction tensor is FP32 [..., 1], before affine scaling.
        counter["tensor_bytes"] += math.prod(inputs[0].shape[:-1]) * 4

    def close(self):
        for handle in self.handles:
            handle.remove()


def gather(value, group):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def aggregate(rank_results, frames):
    """Each device has its own CUDA clock; aggregate local intervals only."""
    result = {"ranks": rank_results}
    for key in (
        "generate_wall_s", "dit_forward_cuda_sum_s",
        "peak_allocated_bytes", "peak_reserved_bytes",
    ):
        result[key] = max(rank[key] for rank in rank_results)
    elapsed = result["generate_wall_s"]
    result["video_per_s"] = 1 / elapsed
    result["generated_frames_per_s"] = frames / elapsed
    result["gpu_seconds_per_video"] = len(rank_results) * elapsed
    return result


def compare_baseline(baseline, report):
    """Require the existing baseline schema and matching measured settings."""
    if not baseline["complete"]:
        raise ValueError("The baseline measurement is incomplete")
    config_keys = (
        "size", "frames", "steps", "seed", "shift", "guide_scale",
        "prompt", "t5_cpu", "offload_model",
    )
    mismatches = [key for key in config_keys if baseline["config"][key] != report["config"][key]]
    for key in ("task", "sample_solver", "negative_prompt"):
        if baseline[key] != report[key]:
            mismatches.append(key)
    environment_keys = (
        "gpu_name", "torch_num_threads", "dit_parameter_dtype",
        "autocast_dtype", "attention_backend", "source_commit", "source_sha256",
        "checkpoint_config", "packages", "torch_cuda",
    )
    for rank in report["environments"]:
        mismatches.extend(
            f"rank{rank['rank']}.{key}" for key in environment_keys
            if baseline["environment"][key] != rank[key]
        )
    if mismatches:
        raise ValueError(f"Baseline settings differ: {', '.join(mismatches)}")
    g1 = baseline["summary"]["generate_wall_s"]["median"]
    d1 = baseline["summary"]["dit_forward_cuda_sum_s"]["median"]
    gp = report["summary"]["generate_wall_s"]["median"]
    dp = report["summary"]["dit_forward_cuda_sum_s"]["median"]
    return {
        "generate_speedup": g1 / gp,
        "generate_parallel_efficiency": g1 / gp / report["tp_size"],
        "dit_speedup": d1 / dp,
        "dit_parallel_efficiency": d1 / dp / report["tp_size"],
        "peak_allocated_reduction": 1 - report["summary"]["peak_allocated_bytes"]["median"]
        / baseline["summary"]["peak_allocated_bytes"]["median"],
        "note": "Metadata matches; host CPU capacity and topology still require review. TP uses rank-zero T5/VAE.",
    }


def main():
    args = parse_args()
    timeout = timedelta(seconds=args.timeout_seconds)
    try:
        device = init_distributed_environment(timeout=timeout)
        initialize_model_parallel(args.tp_size, timeout=timeout)
        group = get_tensor_model_parallel_group()
        rank = dist.get_rank(group)
        logging.basicConfig(
            level=logging.INFO, format=f"%(asctime)s [TP rank {rank}] %(message)s"
        )

        # Bind the device before importing official modules with CUDA defaults.
        import wan
        from pipelines.wan_t2v_tp import WanT2VTP
        from wan.configs import WAN_CONFIGS
        from wan.modules import attention as attention_module
        from wan.utils.utils import cache_video

        wan_dir = args.wan_dir.resolve()
        if Path(wan.__file__).resolve().parent != wan_dir / "wan":
            raise ValueError("PYTHONPATH's wan package does not match --wan-dir")
        actual_commit = command_output(["git", "-C", str(wan_dir), "rev-parse", "HEAD"])
        if actual_commit != SOURCE_COMMIT:
            raise ValueError(f"Wan source commit {actual_commit}; expected {SOURCE_COMMIT}")
        config = WAN_CONFIGS["t2v-1.3B"]
        if attention_module.FLASH_ATTN_3_AVAILABLE:
            backend = "FlashAttention-3 (upstream automatic selection)"
        elif attention_module.FLASH_ATTN_2_AVAILABLE:
            backend = "FlashAttention-2 (upstream automatic selection)"
        else:
            raise RuntimeError("Wan DiT requires FlashAttention")

        metrics_path = args.output_dir / "metrics.json"
        report = None
        if rank == 0:
            if metrics_path.exists():
                raise FileExistsError(f"Use a new --output-dir; results exist: {metrics_path}")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "complete": False,
                "config": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                "task": "t2v-1.3B", "tp_size": args.tp_size,
                "sample_solver": args.sample_solver,
                "negative_prompt": args.negative_prompt or config.sample_neg_prompt,
                "ownership": {"t5": "TP rank 0", "vae": "TP rank 0", "dit": "TP shards on all ranks"},
                "nvidia_smi": command_output([
                    "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
                    "--format=csv",
                ]),
                "gpu_topology": command_output(["nvidia-smi", "topo", "-m"]),
                "host_memory": command_output(["free", "-h"]),
                "measurement_scope": {
                    "generate_wall_s": "Max local wall time from aligned start; CUDA synchronized; includes text/noise broadcasts, T5, sampling, offload, VAE, cleanup and pipeline completion barrier; excludes output checks, result gathering, loading and MP4 encoding",
                    "dit_forward_cuda_sum_s": "Max rank sum of local forward CUDA intervals; includes TP dependencies/waits; not full denoising latency or a kernel-time sum",
                    "memory": "Each rank's full-generate PyTorch allocator peaks and their maxima; not nvidia-smi memory or simultaneous total memory",
                    "communication": "Per-rank DiT SUM AllReduce calls and logical tensor bytes only; excludes pipeline broadcasts, benchmark collectives, bus traffic and communication timing",
                    "hooks": "CUDA event hooks plus lightweight module shape/dtype counters; no profiler or per-forward synchronization",
                },
                "warmup_results": [], "runs": [],
            }

        def save_report():
            metrics_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        if rank == 0:
            save_report()
        dist.barrier(group=group)
        torch.cuda.synchronize(device)
        load_start = time.perf_counter()
        pipeline = WanT2VTP(
            config, args.ckpt_dir, process_group=group, device=device, t5_cpu=args.t5_cpu
        )
        torch.cuda.synchronize(device)
        load_wall_s = time.perf_counter() - load_start
        gpu = torch.cuda.get_device_properties(device)
        project_root = Path(__file__).resolve().parents[1]
        project_files = (
            "pipelines/wan_t2v_tp.py", "models/wan/tp_wan.py",
            "parallel/tp_linear.py", "parallel/tp_attention.py", "parallel/tp_weight_loader.py",
            "runtime/distributed.py", "runtime/parallel_state.py",
            "benchmarks/benchmark_wan_tp2.py", "benchmarks/benchmark_wan_single_gpu.py",
        )
        revision_path = Path("../env-info/wan-model-revision.txt")
        environment = {
            "rank": rank, "device": str(device), "hostname": os.uname().nodename,
            "python": sys.version, "python_executable": sys.executable,
            "cpu_count": os.cpu_count(), "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_cuda": torch.version.cuda,
            "packages": {name: version(name) for name in ("torch", "flash-attn", "triton", "diffusers", "numpy")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": gpu.name, "gpu_total_memory_bytes": gpu.total_memory,
            "gpu_compute_capability": [gpu.major, gpu.minor],
            "source_commit": actual_commit,
            "source_status": command_output(["git", "-C", str(wan_dir), "status", "--short"]),
            "source_sha256": {name: hashlib.sha256((wan_dir / name).read_bytes()).hexdigest()
                              for name in ("wan/text2video.py", "wan/modules/model.py", "wan/modules/attention.py")},
            "project_sha256": {name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
                               for name in project_files},
            "checkpoint_config": json.loads((args.ckpt_dir / "config.json").read_text(encoding="utf-8")),
            "model_revision_record": revision_path.read_text().strip() if revision_path.is_file() else None,
            "attention_backend": backend,
            "dit_parameter_dtype": str(next(pipeline.model.parameters()).dtype),
            "autocast_dtype": str(config.param_dtype),
            "dit_attributes": {name: getattr(pipeline.model, name) for name in (
                "dim", "ffn_dim", "num_heads", "num_layers", "patch_size", "qk_norm"
            )},
            "pipeline_load_wall_s": load_wall_s,
        }
        environments = gather(environment, group)
        if rank == 0:
            report["environments"] = environments
            report["pipeline_load_wall_s"] = max(item["pipeline_load_wall_s"] for item in environments)
            save_report()

        timing = DiTForwardTiming(pipeline.model, torch, expected_calls=2 * args.steps)
        volume = TPAllReduceVolume(pipeline.model)
        width, height = map(int, args.size.split("*"))
        try:
            for index in range(args.warmup + args.runs):
                timing.reset()
                volume.reset()
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                video = pipeline.generate(
                    args.prompt, size=(width, height), frame_num=args.frames,
                    shift=args.shift, sample_solver=args.sample_solver, sampling_steps=args.steps,
                    guide_scale=args.guide_scale, n_prompt=args.negative_prompt, seed=args.seed,
                    offload_model=args.offload_model,
                )
                torch.cuda.synchronize(device)
                wall_s = time.perf_counter() - start
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                forward_ms = timing.elapsed_ms()
                local_result = {
                    "rank": rank, "generate_wall_s": wall_s,
                    "dit_forward_count": timing.count,
                    "dit_forward_cuda_ms": forward_ms,
                    "dit_forward_cuda_ms_summary": summary(forward_ms),
                    "dit_forward_cuda_sum_s": sum(forward_ms) / 1000,
                    "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                    "tp_all_reduce": volume.counts,
                }
                if rank == 0:
                    if tuple(video.shape) != (3, args.frames, height, width):
                        raise RuntimeError(f"Unexpected video shape: {tuple(video.shape)}")
                    if not torch.isfinite(video).all().item():
                        raise RuntimeError("Video contains NaN/Inf")
                    local_result.update(
                        video_shape=list(video.shape), video_dtype=str(video.dtype), video_finite=True
                    )
                rank_results = gather(local_result, group)
                if rank == 0:
                    result = aggregate(rank_results, args.frames)
                    report["warmup_results" if index < args.warmup else "runs"].append(result)
                    save_report()
                    logging.info(
                        "Iteration %d: generate(max)=%.3fs, DiT sum(max)=%.3fs, allocated(max)=%.3fGiB",
                        index + 1, result["generate_wall_s"], result["dit_forward_cuda_sum_s"],
                        result["peak_allocated_bytes"] / 1024**3,
                    )
                    if index == args.warmup + args.runs - 1:
                        saved = cache_video(
                            tensor=video[None], save_file=str(args.output_dir / "last_run.mp4"),
                            fps=config.sample_fps, nrow=1, normalize=True, value_range=(-1, 1), retry=1,
                        )
                        if saved is None:
                            raise RuntimeError("MP4 save failed; partial measurements remain in metrics.json")
                        report["saved_video"] = saved
                del video
        finally:
            timing.close()
            volume.close()

        if rank == 0:
            keys = (
                "generate_wall_s", "dit_forward_cuda_sum_s", "peak_allocated_bytes", "peak_reserved_bytes",
                "video_per_s", "generated_frames_per_s", "gpu_seconds_per_video",
            )
            report["summary"] = {key: summary([run[key] for run in report["runs"]]) for key in keys}
            report["summary_by_rank"] = [
                {"rank": r, **{key: summary([run["ranks"][r][key] for run in report["runs"]])
                               for key in keys[:4]}}
                for r in range(args.tp_size)
            ]
            save_report()
            if args.baseline_metrics is not None:
                baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
                report["baseline_comparison"] = compare_baseline(baseline, report)
            report["complete"] = True
            save_report()
            logging.info("Summary: %s", json.dumps(report["summary"], ensure_ascii=False))
        dist.barrier(group=group)
    finally:
        try:
            destroy_model_parallel()
        finally:
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
