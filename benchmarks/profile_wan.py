"""Profile the existing official/TP2 generation paths on cloud GPUs.

Phase synchronizations perturb timing: this is diagnostic, not a benchmark.
Method observers exist only during this run and delegate to original methods.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import timedelta
import gzip
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from benchmarks.benchmark_wan_single_gpu import PROMPT, SOURCE_COMMIT, command_output
from runtime.distributed import init_distributed_environment, destroy_distributed_environment
from runtime.parallel_state import (
    initialize_model_parallel, get_tensor_model_parallel_group, destroy_model_parallel,
)


def memory(device):
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }


def inventory(pipeline):
    result = {}
    for name, component in (("dit", pipeline.model),
                            ("t5", pipeline.text_encoder), ("vae", pipeline.vae)):
        if component is None:
            continue
        module = component if name == "dit" else component.model
        totals = {}
        for tensor in (*module.parameters(), *module.buffers()):
            key = f"{tensor.device}/{tensor.dtype}"
            totals[key] = totals.get(key, 0) + tensor.numel() * tensor.element_size()
        result[name] = totals
    return result


def kernel_summary(trace_path):
    """Use CUDA kernel events, not overlapping parent operator totals."""
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"
               and event.get("ph") == "X"]
    if not kernels:
        raise RuntimeError("No CUDA kernels in profiler trace; check CUPTI/profiler availability")
    buckets = {}
    for event in kernels:
        name = event["name"].lower()
        category = ("nccl" if "nccl" in name else
                    "attention" if any(term in name for term in ("flash", "fmha", "attention")) else
                    "gemm" if any(term in name for term in ("gemm", "cublas", "cutlass", "matmul")) else "other")
        bucket = buckets.setdefault(category, dict(calls=0, duration_sum_us=0))
        bucket["calls"] += 1
        bucket["duration_sum_us"] += event["dur"]
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in kernels)
    start, end = intervals[0]
    covered = 0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    covered += end - start
    span = max(b for _, b in intervals) - intervals[0][0]
    return dict(
        categories=buckets, kernel_count=len(kernels), kernel_span_us=span,
        recorded_kernel_union_us=covered, uncovered_span_us=span - covered,
        note="Categories inferred from kernel names. Duration sums may overlap across streams; "
             "union is recorded kernel coverage, uncovered span is not automatically CPU overhead. "
             "NCCL kernel duration may include peer waits, not only data transfer.",
    )


class PhaseObserver:
    """Partition generate at its existing device-transfer/decode boundaries."""

    def __init__(self, pipeline, device, report, save):
        self.pipeline, self.device, self.report, self.save = pipeline, device, report, save
        self.current = None
        self.replacements = []

    def begin(self, name):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.current = dict(name=name, start=memory(self.device))
        self.scope = record_function("phase/" + name)
        self.scope.__enter__()
        self.start = time.perf_counter()

    def end(self):
        torch.cuda.synchronize(self.device)
        self.current.update(
            wall_s=time.perf_counter() - self.start, end=memory(self.device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
        )
        self.scope.__exit__(None, None, None)
        self.report["phases"].append(self.current)
        self.save()
        print(f"[rank {self.report['rank']}] {self.current['name']}: "
              f"{self.current['wall_s']:.3f}s, "
              f"allocated peak={self.current['peak_allocated_bytes'] / 2**30:.3f} GiB", flush=True)
        self.current = None

    def replace(self, owner, name, function):
        # Restore the original instance dictionary, not a copied bound method.
        self.replacements.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, function)

    @contextmanager
    def observe(self):
        model = self.pipeline.model
        original_to, original_cpu = model.to, model.cpu

        def to(*args, **kwargs):
            # Official Wan calls .to(device) every step; only the first is restore.
            if self.current["name"] != "prepare_inputs":
                return original_to(*args, **kwargs)
            self.end()
            self.begin("restore_dit")
            result = original_to(*args, **kwargs)
            self.end()
            self.begin("sampling")
            return result

        def cpu(*args, **kwargs):
            self.end()
            self.begin("offload_dit")
            result = original_cpu(*args, **kwargs)
            self.end()
            self.begin("after_offload")
            return result

        self.replace(model, "to", to)
        self.replace(model, "cpu", cpu)
        if self.pipeline.vae is not None:
            original_decode = self.pipeline.vae.decode

            def decode(*args, **kwargs):
                self.end()
                self.begin("vae_decode")
                result = original_decode(*args, **kwargs)
                self.end()
                self.begin("cleanup_and_completion_wait")
                return result

            self.replace(self.pipeline.vae, "decode", decode)
        self.begin("prepare_inputs")
        try:
            yield
            self.end()
        finally:
            if self.current is not None:
                self.scope.__exit__(None, None, None)
                self.current = None
            for owner, name, existed, value in reversed(self.replacements):
                if existed:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)


def module_ranges(model):
    """CPU range attribution without synchronizing individual modules."""
    handles = []
    for name, module in model.named_modules():
        if not (name.endswith(("self_attn", "cross_attn", "ffn"))
                or module.__class__.__name__ in ("ColumnParallelLinear", "RowParallelLinear", "TPWanRMSNorm")):
            continue
        scopes = []

        def before(module, inputs, label=name, stack=scopes):
            scope = record_function("module/" + label)
            scope.__enter__()
            stack.append(scope)

        def after(module, inputs, output, stack=scopes):
            stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))
    return handles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), required=True,
                        help="1 profiles the original official pipeline; 2 profiles ours.")
    parser.add_argument("--wan-dir", type=Path, default=Path("../Wan2.1"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("../models/Wan2.1-T2V-1.3B"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    handles = []
    completed = False
    try:
        timeout = timedelta(seconds=3600)
        device = init_distributed_environment(timeout=timeout)
        initialize_model_parallel(args.tp_size, timeout=timeout)
        group = get_tensor_model_parallel_group()
        rank = dist.get_rank(group)
        import wan
        from wan.configs import WAN_CONFIGS
        if Path(wan.__file__).resolve().parent != args.wan_dir.resolve() / "wan":
            raise ValueError("Imported Wan does not match --wan-dir")
        if command_output(["git", "-C", str(args.wan_dir), "rev-parse", "HEAD"]) != SOURCE_COMMIT:
            raise ValueError("Official Wan commit differs from the fixed reference")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / f"rank_{rank}.json"
        if report_path.exists():
            raise FileExistsError(f"Use a new output directory: {report_path}")
        report = dict(
            complete=False, rank=rank, tp_size=args.tp_size,
            implementation="official" if args.tp_size == 1 else "ours_tp2",
            config=dict(size=[832, 480], frames=81, steps=50, seed=42,
                        solver="unipc", shift=8, cfg=6, t5_cpu=True, offload=True, prompt=PROMPT),
            environment=dict(torch=torch.__version__, device=str(device),
                             gpu=torch.cuda.get_device_name(device), source_commit=SOURCE_COMMIT,
                             topology=command_output(["nvidia-smi", "topo", "-m"])),
            scope=dict(
                timing="Diagnostic only: synchronized phase boundaries and profiler perturb timings.",
                memory="PyTorch allocator per-phase peaks include live allocations at phase start; not nvidia-smi.",
                inventory="Parameters and registered buffers only; excludes activations, unregistered tensors and external workspaces.",
                prepare_inputs="CPU T5/input preparation and, on TP2, input broadcasts or their waits.",
                sampling="Full 50-step denoising loop, including CFG and scheduler, excluding first DiT restore/offload.",
                after_offload="Rank 0: empty_cache before VAE. Rank 1: empty_cache, cleanup and completion wait; no VAE.",
                trace="Two sampling steps after 5 skipped and 1 profiler warmup step; not the entire video.",
                operators="Nested inclusive operator times cannot be summed into wall time or critical-path percentages.",
            ), phases=[],
        )

        def save():
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        save()
        cfg = WAN_CONFIGS["t2v-1.3B"]
        if args.tp_size == 1:
            pipeline = wan.WanT2V(config=cfg, checkpoint_dir=str(args.ckpt_dir), device_id=device.index,
                                  rank=0, t5_cpu=True, use_usp=False, t5_fsdp=False, dit_fsdp=False)
        else:
            from pipelines.wan_t2v_tp import WanT2VTP
            pipeline = WanT2VTP(cfg, args.ckpt_dir, process_group=group, device=device, t5_cpu=True)
        torch.cuda.synchronize(device)
        report["after_load"] = dict(memory=memory(device), components=inventory(pipeline))
        report["environment"].update(
            dit_parameter_dtype=str(next(pipeline.model.parameters()).dtype),
            autocast_dtype=str(cfg.param_dtype), torch_num_threads=torch.get_num_threads(),
        )
        save()
        kwargs = dict(size=(832, 480), frame_num=81, sampling_steps=50, seed=42,
                      sample_solver="unipc", shift=8, guide_scale=6, offload_model=True)
        print(f"[rank {rank}] Warmup: full 81-frame generation", flush=True)
        video = pipeline.generate(PROMPT, **kwargs)
        del video
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        report["after_warmup"] = dict(memory=memory(device), components=inventory(pipeline))
        save()

        def trace_ready(profiler):
            trace_path = args.output_dir / f"rank_{rank}.trace.json.gz"
            profiler.export_chrome_trace(str(trace_path))
            averages = profiler.key_averages()
            (args.output_dir / f"rank_{rank}.operators.txt").write_text(
                averages.table(sort_by="self_device_time_total", row_limit=80), encoding="utf-8")
            rows = [dict(name=event.key, calls=event.count, device_type=str(event.device_type),
                         self_cpu_us=event.self_cpu_time_total, cpu_us=event.cpu_time_total,
                         self_device_us=event.self_device_time_total, device_us=event.device_time_total,
                         self_device_memory_bytes=event.self_device_memory_usage) for event in averages]
            (args.output_dir / f"rank_{rank}.operators.json").write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            report["trace_saved"] = str(trace_path)
            report["kernel_summary"] = kernel_summary(trace_path)
            save()

        handles.extend(module_ranges(pipeline.model))
        count = 0
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=5, warmup=1, active=2, repeat=1),
            on_trace_ready=trace_ready, record_shapes=False, with_stack=False, profile_memory=True,
        ) as profiler:
            def step_after(module, inputs, output):
                nonlocal count
                count += 1
                if count % 2 == 0:
                    profiler.step()

            handles.append(pipeline.model.register_forward_hook(step_after))
            observer = PhaseObserver(pipeline, device, report, save)
            print(f"[rank {rank}] Diagnostic generation: 81 frames / 50 steps", flush=True)
            with observer.observe():
                video = pipeline.generate(PROMPT, **kwargs)
        for handle in handles:
            handle.remove()
        handles.clear()
        report["dit_forward_count"] = count
        if count != 100 or "trace_saved" not in report:
            raise RuntimeError("Expected 100 forwards and a completed profiler trace")
        if rank == 0:
            report["video_shape"] = list(video.shape)
            report["video_finite"] = bool(torch.isfinite(video).all().item())
            if tuple(video.shape) != (3, 81, 480, 832) or not report["video_finite"]:
                raise RuntimeError("Invalid diagnostic video")
        report["largest_allocated_phase"] = max(report["phases"], key=lambda p: p["peak_allocated_bytes"])["name"]
        report["after_generate"] = dict(memory=memory(device), components=inventory(pipeline))
        save()
        del video
        torch.cuda.synchronize(device)
        report["after_video_release"] = memory(device)
        dist.barrier(group=group)
        completed = True
    finally:
        for handle in handles:
            handle.remove()
        try:
            destroy_model_parallel()
        finally:
            destroy_distributed_environment()
    if completed:
        report["complete"] = True
        save()
        print(f"[rank {rank}] Completed: {report_path}; largest peak: {report['largest_allocated_phase']}", flush=True)


if __name__ == "__main__":
    main()
