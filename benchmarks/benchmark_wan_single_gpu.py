"""Measure the original Wan2.1-T2V-1.3B pipeline on one cloud GPU.

Run from the wan_parallel project root. The upstream forward and generation logic
remain the baseline; hooks only record CUDA events around DiT calls.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version


SOURCE_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-dir", type=Path, default=Path("../Wan2.1"))
    parser.add_argument(
        "--ckpt-dir", type=Path, default=Path("../models/Wan2.1-T2V-1.3B")
    )
    parser.add_argument("--size", choices=("832*480", "480*832"), default="832*480")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--offload-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error("--frames 必须是正数且满足 4n+1")
    if args.steps < 1 or args.warmup < 1 or args.runs < 1 or args.seed < 0:
        parser.error("steps、warmup、runs 必须至少为 1，seed 必须非负")
    return args


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"


def summary(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


class DiTForwardTiming:
    """Reuse warmed-up event pairs; never synchronize inside a forward hook."""

    def __init__(self, model, torch_module, expected_calls):
        self.pairs = [
            (
                torch_module.cuda.Event(enable_timing=True),
                torch_module.cuda.Event(enable_timing=True),
            )
            for _ in range(expected_calls)
        ]
        self.count = 0
        self.handles = [
            model.register_forward_pre_hook(self.before),
            model.register_forward_hook(self.after),
        ]

    def before(self, module, inputs):
        if self.count >= len(self.pairs):
            raise RuntimeError("DiT forward 次数超过 2 * steps，请检查原始采样路径")
        self.pairs[self.count][0].record()

    def after(self, module, inputs, output):
        self.pairs[self.count][1].record()
        self.count += 1

    def reset(self):
        self.count = 0

    def elapsed_ms(self):
        if self.count != len(self.pairs):
            raise RuntimeError(f"预期 {len(self.pairs)} 次 DiT forward，实际 {self.count}")
        return [start.elapsed_time(end) for start, end in self.pairs]

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    wan_dir = args.wan_dir.resolve()
    actual_commit = command_output(["git", "-C", str(wan_dir), "rev-parse", "HEAD"])
    if actual_commit != SOURCE_COMMIT:
        raise SystemExit(f"Wan 源码版本不匹配：{actual_commit}；预期 {SOURCE_COMMIT}")
    if not (args.ckpt_dir / "config.json").is_file():
        raise SystemExit(f"模型目录缺少 config.json：{args.ckpt_dir}")
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists():
        raise SystemExit(f"结果已存在，请选择新的 --output-dir：{metrics_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(wan_dir))
    import torch
    from wan import WanT2V
    from wan.configs import WAN_CONFIGS
    from wan.modules import attention as attention_module
    from wan.utils.utils import cache_video

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("请通过 CUDA_VISIBLE_DEVICES=0 暴露且只暴露一张可用 GPU")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("当前 GPU 不支持本 baseline 使用的 BF16")
    cfg = WAN_CONFIGS["t2v-1.3B"]
    gpu = torch.cuda.get_device_properties(0)
    package_versions = {}
    for package in ("torch", "flash-attn", "triton", "diffusers", "numpy"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    if attention_module.FLASH_ATTN_3_AVAILABLE:
        attention_backend = "FlashAttention-3 (upstream automatic selection)"
    elif attention_module.FLASH_ATTN_2_AVAILABLE:
        attention_backend = "FlashAttention-2 (upstream automatic selection)"
    else:
        raise SystemExit("原始 WanModel 需要可用的 FlashAttention")

    report = {
        "complete": False,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "task": "t2v-1.3B",
        "sample_solver": "unipc",
        "negative_prompt": cfg.sample_neg_prompt,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "host_memory": command_output(["free", "-h"]),
            "packages": package_versions,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": gpu.name,
            "gpu_total_memory_bytes": gpu.total_memory,
            "gpu_compute_capability": [gpu.major, gpu.minor],
            "nvidia_smi": command_output([
                "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv",
            ]),
            "source_commit": actual_commit,
            "source_status": command_output(["git", "-C", str(wan_dir), "status", "--short"]),
            "source_sha256": {
                name: hashlib.sha256((wan_dir / name).read_bytes()).hexdigest()
                for name in ("wan/text2video.py", "wan/modules/model.py", "wan/modules/attention.py")
            },
            "checkpoint_config": json.loads((args.ckpt_dir / "config.json").read_text()),
            "model_revision_record": (
                Path("../env-info/wan-model-revision.txt").read_text().strip()
                if Path("../env-info/wan-model-revision.txt").is_file() else None
            ),
            "attention_backend": attention_backend,
            "autocast_dtype": str(cfg.param_dtype),
        },
        "measurement_scope": {
            "generate_wall_s": "CUDA-synchronized WanT2V.generate: T5, sampling, offload, VAE, cleanup; excludes pipeline loading and MP4 encoding",
            "dit_forward_cuda_ms": "CUDA-event interval around each original DiT forward; includes stream idle/host submission gaps, not a kernel-time sum or full denoising latency",
            "peak_memory": "PyTorch allocator peaks per generate call, including live allocations at the start; not total device usage",
        },
        "warmup_results": [],
        "runs": [],
    }

    def save_report():
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_report()
    torch.cuda.synchronize()
    load_start = time.perf_counter()
    pipeline = WanT2V(
        config=cfg, checkpoint_dir=str(args.ckpt_dir), device_id=0, rank=0,
        t5_cpu=args.t5_cpu, use_usp=False, t5_fsdp=False, dit_fsdp=False,
    )
    torch.cuda.synchronize()
    report["pipeline_load_wall_s"] = time.perf_counter() - load_start
    report["environment"]["dit_parameter_dtype"] = str(next(pipeline.model.parameters()).dtype)
    report["environment"]["dit_attributes"] = {
        name: getattr(pipeline.model, name)
        for name in ("dim", "ffn_dim", "num_heads", "num_layers", "patch_size", "qk_norm")
    }
    save_report()

    timing = DiTForwardTiming(pipeline.model, torch, expected_calls=2 * args.steps)
    width, height = map(int, args.size.split("*"))
    try:
        for index in range(args.warmup + args.runs):
            is_warmup = index < args.warmup
            label = f"warmup {index + 1}" if is_warmup else f"run {index - args.warmup + 1}"
            logging.info("开始 %s", label)
            timing.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            video = pipeline.generate(
                args.prompt, size=(width, height), frame_num=args.frames,
                shift=args.shift, sample_solver="unipc", sampling_steps=args.steps,
                guide_scale=args.guide_scale, seed=args.seed,
                offload_model=args.offload_model,
            )
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - start
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            forward_ms = timing.elapsed_ms()
            if tuple(video.shape) != (3, args.frames, height, width):
                raise RuntimeError(f"视频输出形状不符合预期：{tuple(video.shape)}")
            if not torch.isfinite(video).all().item():
                raise RuntimeError("视频输出包含 NaN/Inf")
            result = {
                "generate_wall_s": wall_s,
                "dit_forward_count": timing.count,
                "dit_forward_cuda_ms": forward_ms,
                "dit_forward_cuda_ms_summary": summary(forward_ms),
                "dit_forward_cuda_sum_s": sum(forward_ms) / 1000,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "video_shape": list(video.shape),
                "video_dtype": str(video.dtype),
                "video_finite": True,
            }
            report["warmup_results" if is_warmup else "runs"].append(result)
            save_report()
            logging.info(
                "%s: generate=%.3fs, DiT CUDA sum=%.3fs, peak allocated=%.3fGiB",
                label, wall_s, result["dit_forward_cuda_sum_s"], allocated / 1024**3,
            )
            if index == args.warmup + args.runs - 1:
                video_path = args.output_dir / "last_run.mp4"
                saved = cache_video(
                    tensor=video[None], save_file=str(video_path), fps=cfg.sample_fps,
                    nrow=1, normalize=True, value_range=(-1, 1),
                )
                if saved is None:
                    raise RuntimeError("MP4 保存失败，已完成的计时结果保留在 metrics.json")
                report["saved_video"] = str(video_path)
            del video
    finally:
        timing.close()

    report["summary"] = {
        key: summary([run[key] for run in report["runs"]])
        for key in ("generate_wall_s", "dit_forward_cuda_sum_s", "peak_allocated_bytes", "peak_reserved_bytes")
    }
    report["complete"] = True
    save_report()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"完整结果：{metrics_path}")


if __name__ == "__main__":
    main()
