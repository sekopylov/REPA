import argparse
import copy
from copy import deepcopy
import logging
import os
import time
from pathlib import Path
from collections import OrderedDict
import json
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
try:
    import wandb
except ImportError:
    wandb = None
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD  = (0.26862954, 0.26130258, 0.27577711)

# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def augment_flip_only(raw_image: torch.Tensor, x: torch.Tensor):
    B = raw_image.shape[0]
    flip_mask = torch.rand(B, device=raw_image.device) < 0.5
    flip_mask_view = flip_mask.view(B, 1, 1, 1)
    raw_aug = torch.where(flip_mask_view, raw_image.flip(-1), raw_image)
    x_aug   = torch.where(flip_mask_view, x.flip(-1), x)
    return raw_aug, x_aug


@torch.no_grad()
def augment_vae(raw_image: torch.Tensor, vae, latents_scale, latents_bias, resolution: int):
    B      = raw_image.shape[0]
    device = raw_image.device
    flip_mask  = torch.rand(B, device=device) < 0.5
    img_float  = raw_image.float() / 255.0

    raw_aug_list = []
    for i in range(B):
        img_i = img_float[i]
        top, left, height, width = T.RandomResizedCrop.get_params(
            img_i, scale=(0.8, 1.0), ratio=(3/4, 4/3))
        img_i = TF.resized_crop(img_i, top, left, height, width,
                                size=[resolution, resolution],
                                interpolation=T.InterpolationMode.BICUBIC,
                                antialias=True)
        if flip_mask[i]:
            img_i = img_i.flip(-1)
        raw_aug_list.append(img_i)

    raw_aug_float = torch.stack(raw_aug_list, dim=0)
    vae_input     = raw_aug_float * 2.0 - 1.0
    posterior     = vae.encode(vae_input).latent_dist
    z             = posterior.sample()
    x_aug         = z * latents_scale + latents_bias

    brightness, contrast, saturation, hue = 0.2, 0.2, 0.2, 0.05
    raw_aug_jitter = []
    for i in range(B):
        img_i     = raw_aug_float[i]
        fn_order  = torch.randperm(4)
        for fn_id in fn_order:
            if fn_id == 0:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * brightness
                img_i  = TF.adjust_brightness(img_i, max(0, factor))
            elif fn_id == 1:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * contrast
                img_i  = TF.adjust_contrast(img_i, max(0, factor))
            elif fn_id == 2:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * saturation
                img_i  = TF.adjust_saturation(img_i, max(0, factor))
            else:
                factor = (torch.rand(1).item() * 2 - 1) * hue
                img_i  = TF.adjust_hue(img_i, max(-0.5, min(0.5, factor)))
        raw_aug_jitter.append(img_i)

    raw_aug_float = torch.stack(raw_aug_jitter, dim=0)
    raw_aug       = (raw_aug_float * 255.0).clamp(0, 255).to(torch.uint8)
    return raw_aug, x_aug


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x    = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x    = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    # Peel DDP wrapper, then torch.compile wrapper — both optional
    raw = getattr(model, 'module', model)
    raw = getattr(raw, '_orig_mod', raw)
    for name, param in raw.named_parameters():
        ema_params[name].lerp_(param.data, 1.0 - decay)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def uses_repa(args):
    if args.enc_type is None:
        return False
    return args.enc_type.lower() not in {"none", "no-repa", "baseline"} and args.proj_coeff > 0


def should_log(args):
    return args.report_to.lower() != "none"


def uses_wandb(args):
    return args.report_to.lower() == "wandb"


#################################################################################
# Training Loop                                                                 #
#################################################################################

def main(args):
    logging_dir = Path(args.output_dir, args.exp_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    log_with   = None if not should_log(args) else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    logger = logging.getLogger(__name__)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir       = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    assert args.resolution % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.resolution // 8

    use_repa = uses_repa(args)
    if use_repa:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution)
    else:
        encoders, encoder_types, architectures = [], [], []

    z_dims       = [encoder.embed_dim for encoder in encoders]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        class_dropout_prob=args.cfg_prob,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )
    model = model.to(device)
    ema   = deepcopy(model).to(device)   # plain copy BEFORE any wrapping
    vae   = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    requires_grad(vae, False)

    latents_scale = torch.tensor([0.18215]*4).view(1, 4, 1, 1).to(device)
    latents_bias  = torch.tensor([0.]*4).view(1, 4, 1, 1).to(device)

    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        div_coeff=args.div_coeff,
    )

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"CFG dropout: {args.cfg_prob}")
        if args.augment:
            logger.info(f"Augmentation mode: {args.augment_mode}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset    = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + '.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    if accelerator.is_main_process and should_log(args):
        init_kwargs = {}
        if uses_wandb(args):
            init_kwargs["wandb"] = {"name": args.exp_name}
        accelerator.init_trackers(
            project_name="REPA",
            config=vars(copy.deepcopy(args)),
            init_kwargs=init_kwargs,
        )

    _log_every = getattr(args, 'log_every', 100)

    # Fixed sample batch for qualitative logging (rank-local, no gather needed here)
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias)
    ys    = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    xT    = torch.randn(ys.size(0), 4, latent_size, latent_size, device=device)

    grad_norm       = torch.tensor(0.0, device=device)
    initial_step    = global_step
    train_start     = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            step_start = time.perf_counter()
            raw_image  = raw_image.to(device)
            x          = x.squeeze(dim=1).to(device)
            y          = y.to(device)

            # Augmentation
            if args.augment:
                if args.augment_mode == 'vae':
                    with torch.no_grad():
                        raw_image, x = augment_vae(raw_image, vae, latents_scale, latents_bias, args.resolution)
                else:  # flip_only
                    raw_image, x = augment_flip_only(raw_image, x)

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            labels = y  # LabelEmbedder handles cfg dropout internally

            # Teacher forward pass
            teacher_start = time.perf_counter()
            zs = []
            if use_repa:
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_proc = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_proc)
                        if 'mocov3' in encoder_type: z = z[:, 1:]
                        if 'dinov2' in encoder_type:  z = z['x_norm_patchtokens']
                        zs.append(z)
            teacher_time = time.perf_counter() - teacher_start

            # Model forward + backward
            with accelerator.accumulate(model):
                model_kwargs   = dict(y=labels)
                denoising_loss, proj_loss, div_loss = loss_fn(model, x, model_kwargs, zs=zs)
                denoising_mean = denoising_loss.mean()
                proj_mean      = proj_loss.mean()
                div_mean       = div_loss.mean()
                total_loss     = denoising_mean + proj_mean * args.proj_coeff + div_mean * args.div_coeff
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                update_ema(ema, model)
                global_step += 1

            # Checkpointing
            if accelerator.sync_gradients and global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema":   ema.state_dict(),
                        "opt":   optimizer.state_dict(),
                        "args":  args,
                        "steps": global_step,
                    }
                    ckpt_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")

            # Qualitative sampling
            if accelerator.sync_gradients and (
                    (args.sample_at_step_one and global_step == 1) or
                    (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                ema.eval()
                with torch.no_grad():
                    samples = euler_sampler(
                        ema, xT, ys, num_steps=50,
                        cfg_scale=args.sample_cfg_scale,
                        guidance_low=0., guidance_high=1.,
                        path_type=args.path_type, heun=False,
                    ).to(torch.float32)
                    samples    = vae.decode((samples - latents_bias) / latents_scale).sample
                    gt_decoded = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples    = (samples + 1) / 2.
                    gt_decoded = (gt_decoded + 1) / 2.
                # Gather is a collective — all ranks must call it
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_out      = accelerator.gather(gt_decoded.to(torch.float32))
                if accelerator.is_main_process and uses_wandb(args) and wandb is not None:
                    accelerator.log({
                        "samples":    wandb.Image(array2grid(out_samples)),
                        "gt_samples": wandb.Image(array2grid(gt_out)),
                    })
                logger.info("Generating EMA samples done.")

            step_time = time.perf_counter() - step_start

            # Logging — ALL ranks gather, only main rank prints/logs
            # RULE: accelerator.gather is a collective → must be called by every rank unconditionally
            total_val    = accelerator.gather(total_loss).mean().detach().item()
            denoise_val  = accelerator.gather(denoising_mean).mean().detach().item()
            proj_val     = accelerator.gather(proj_mean).mean().detach().item()
            div_val      = accelerator.gather(div_mean).mean().detach().item()
            gn_val       = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

            if accelerator.is_main_process and global_step % _log_every == 0:
                avg_step  = (time.perf_counter() - train_start) / max(1, global_step - initial_step)
                eta_str   = str(datetime.timedelta(seconds=int((args.max_train_steps - global_step) * avg_step)))
                print(f"step {global_step:6d}/{args.max_train_steps} "
                      f"diff {denoise_val:.4f} proj {proj_val:.4f} "
                      f"div {div_val:.4f} gn {gn_val:.3f} "
                      f"loss {total_val:.4f} {step_time:.2f}s/step ETA {eta_str}", flush=True)
                if should_log(args):
                    accelerator.log({
                        "loss/total":     total_val,
                        "loss/denoising": denoise_val,
                        "loss/proj":      proj_val,
                        "loss/div":       div_val,
                        "grad_norm":      gn_val,
                        "step_time":      step_time,
                        "teacher_time":   teacher_time,
                    }, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output-dir",   type=str,  default="exps")
    parser.add_argument("--exp-name",     type=str,  required=True)
    parser.add_argument("--logging-dir",  type=str,  default="logs")
    parser.add_argument("--report-to",    type=str,  default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--sample-at-step-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every",    type=int,  default=100)
    parser.add_argument("--resume-step",  type=int,  default=0)

    parser.add_argument("--model",         type=str)
    parser.add_argument("--num-classes",   type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",    action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--data-dir",   type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment-mode", type=str, default="vae", choices=["vae", "flip_only"])

    parser.add_argument("--allow-tf32",       action="store_true")
    parser.add_argument("--mixed-precision",  type=str, default="fp16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--epochs",                      type=int,   default=1400)
    parser.add_argument("--max-train-steps",             type=int,   default=400000)
    parser.add_argument("--checkpointing-steps",         type=int,   default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int,   default=1)
    parser.add_argument("--learning-rate",               type=float, default=1e-4)
    parser.add_argument("--adam-beta1",                  type=float, default=0.9)
    parser.add_argument("--adam-beta2",                  type=float, default=0.999)
    parser.add_argument("--adam-weight-decay",           type=float, default=0.)
    parser.add_argument("--adam-epsilon",                type=float, default=1e-08)
    parser.add_argument("--max-grad-norm",               type=float, default=1.0)

    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--path-type",   type=str,   default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction",  type=str,   default="v",      choices=["v"])
    parser.add_argument("--cfg-prob",    type=float, default=0.1)
    parser.add_argument("--enc-type",    type=str,   default="dinov2-vit-b")
    parser.add_argument("--proj-coeff",  type=float, default=0.5)
    parser.add_argument("--weighting",   type=str,   default="uniform")
    parser.add_argument("--legacy",      action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--div-coeff",   type=float, default=0.0)
    parser.add_argument("--sample-cfg-scale", type=float, default=2.0)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)