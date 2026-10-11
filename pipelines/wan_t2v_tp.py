"""Generate Wan T2V videos with a directly constructed tensor-parallel DiT."""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import sys
from os import PathLike
from pathlib import Path

import torch
import torch.distributed as dist
from easydict import EasyDict
from tqdm import tqdm

from models.wan.tp_wan import create_wan_tp_model, load_wan_tp_weights
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2VTP:
    """Own rank-zero T5/VAE and a rank-local TP DiT for T2V inference.

    The caller initializes WORLD and TP, binds the CUDA device, and keeps the
    supplied TP group alive for the pipeline's lifetime. All TP ranks construct
    this pipeline using the same checkpoint and configuration, then call
    generate together with identical sampling settings. Rank zero encodes text
    and decodes video; every rank runs both TP forwards and its own scheduler.
    """

    def __init__(
        self,
        config: EasyDict,
        checkpoint_dir: str | PathLike[str],
        *,
        process_group: dist.ProcessGroup,
        device: torch.device,
        t5_cpu: bool = True,
    ) -> None:
        """Load official T5/VAE on TP rank zero and TP DiT on every rank.

        ``config`` is the official WAN_CONFIGS pipeline configuration. DiT
        architecture comes separately from checkpoint/config.json. T5 starts on
        CPU as in the official pipeline; t5_cpu selects its later encoding
        placement. DiT parameters are FP32 to match the BF16-autocast baseline.
        """
        self.config = config
        self.process_group = process_group
        self.device = device
        self.t5_cpu = t5_cpu
        checkpoint_dir = Path(checkpoint_dir)

        self.text_encoder: T5EncoderModel | None = None
        self.vae: WanVAE | None = None
        if dist.get_rank(process_group) == 0:
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=str(checkpoint_dir / config.t5_checkpoint),
                tokenizer_path=str(checkpoint_dir / config.t5_tokenizer),
            )
            # Keep the official VAE dtype and latent scaling independent of DiT.
            self.vae = WanVAE(
                vae_pth=str(checkpoint_dir / config.vae_checkpoint),
                device=device,
            )

        with (checkpoint_dir / "config.json").open(encoding="utf-8") as stream:
            dit_config = json.load(stream)

        logging.info("Creating TP WanModel from %s", checkpoint_dir)
        self.model = create_wan_tp_model(
            dit_config,
            process_group,
            device=device,
            dtype=torch.float32,
        )
        load_wan_tp_weights(
            self.model,
            checkpoint_dir / "diffusion_pytorch_model.safetensors",
        )
        self.model.eval().requires_grad_(False)

    def _broadcast_context(self, context: torch.Tensor | None) -> torch.Tensor:
        """Broadcast one unpadded official T5 output from TP rank zero.

        T5's final norm returns config.t5_dtype; its hidden width matches the
        DiT text_dim. Only the token length varies between prompts. Receivers
        need that length before allocating the NCCL broadcast destination.
        """
        group = self.process_group
        src = dist.get_global_rank(group, 0)
        if dist.get_rank(group) == 0:
            context = context.to(self.device).contiguous()
            length = torch.tensor([context.shape[0]], device=self.device, dtype=torch.long)
        else:
            length = torch.empty(1, device=self.device, dtype=torch.long)
        dist.broadcast(length, src=src, group=group)

        if dist.get_rank(group) != 0:
            context = torch.empty(
                (int(length.item()), self.model.text_dim),
                device=self.device,
                dtype=self.config.t5_dtype,
            )
        dist.broadcast(context, src=src, group=group)
        return context

    @torch.no_grad()
    def generate(
        self,
        input_prompt: str,
        size: tuple[int, int] = (1280, 720),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ) -> torch.Tensor | None:
        """Sample with official Wan solver semantics and rank-zero inputs.

        All TP ranks call together with identical size (width, height), frame
        count, solver, steps, shift, guidance and offload settings. Rank zero's
        prompts and seed determine the shared inputs. Returns [3, F, H, W] on
        rank zero and None elsewhere, after the common completion barrier.
        Defaults follow official WanT2V.generate, except T5 placement is chosen
        at construction. Formal baseline settings must be passed explicitly.
        """
        group = self.process_group
        rank = dist.get_rank(group)
        src = dist.get_global_rank(group, 0)

        # T2V latent channels come from DiT, so nonzero ranks need no VAE.
        vae_stride = self.config.vae_stride
        patch_size = self.model.patch_size
        target_shape = (
            self.model.in_dim,
            (frame_num - 1) // vae_stride[0] + 1,
            size[1] // vae_stride[1],
            size[0] // vae_stride[2],
        )
        seq_len = math.ceil(
            target_shape[2] * target_shape[3]
            / (patch_size[1] * patch_size[2]) * target_shape[1]
        )

        # A new scheduler owns this call's history on each rank.
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = scheduler.timesteps
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(scheduler, device=self.device, sigmas=sigmas)
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")

        seed_tensor = torch.empty(1, device=self.device, dtype=torch.long)
        if rank == 0:
            seed_tensor.fill_(seed if seed >= 0 else random.randint(0, sys.maxsize))
        dist.broadcast(seed_tensor, src=src, group=group)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(int(seed_tensor.item()))

        context = context_null = None
        if rank == 0:
            if n_prompt == "":
                n_prompt = self.config.sample_neg_prompt
            text_device = torch.device("cpu") if self.t5_cpu else self.device
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], text_device)[0]
            context_null = self.text_encoder([n_prompt], text_device)[0]
            if not self.t5_cpu and offload_model:
                self.text_encoder.model.cpu()

        context = [self._broadcast_context(context)]
        context_null = [self._broadcast_context(context_null)]
        if rank == 0:
            noise = torch.randn(
                target_shape, dtype=torch.float32, device=self.device, generator=seed_g
            )
        else:
            noise = torch.empty(target_shape, dtype=torch.float32, device=self.device)
        dist.broadcast(noise, src=src, group=group)

        # A preceding generate may have offloaded DiT. Restore once per call.
        self.model.to(self.device)
        video = None
        with torch.autocast(device_type="cuda", dtype=self.config.param_dtype):
            latents = [noise]
            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}
            for t in tqdm(timesteps, disable=rank != 0):
                timestep = torch.stack([t])
                noise_pred_cond = self.model(latents, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latents, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                next_latent = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents = [next_latent.squeeze(0)]

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if rank == 0:
                video = self.vae.decode(latents)[0]

        del noise, latents, scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize(self.device)
        # Normal completion only: errors propagate to the owning run entry.
        dist.barrier(group=group)
        return video
