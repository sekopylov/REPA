import argparse
import copy
from copy import deepcopy
import logging
import os
import time
from pathlib import Path
from collections import OrderedDict
import json

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
    """
    Fallback augmentation: random horizontal flip applied consistently to both
    the raw uint8 image (for the teacher encoder) and the precomputed VAE latent
    (for the diffusion model).  The VAE encoder is spatially aligned with the
    image, so flipping the 4-channel latent on dim=-1 is equivalent to flipping
    the corresponding image.

    Args:
        raw_image: (B, C, H, W) uint8 tensor on any device
        x:         (B, 4, H//8, W//8) float latent tensor on any device

    Returns:
        raw_image_aug, x_aug — same shapes, consistently flipped
    """
    B = raw_image.shape[0]
    flip_mask = torch.rand(B, device=raw_image.device) < 0.5   # (B,) bool
    # Flip along width dimension (dim=-1) for samples where flip_mask is True
    raw_aug = raw_image.clone()
    x_aug   = x.clone()
    for i in range(B):
        if flip_mask[i]:
            raw_aug[i] = raw_image[i].flip(-1)
            x_aug[i]   = x[i].flip(-1)
    return raw_aug, x_aug


@torch.no_grad()
def augment_vae(raw_image: torch.Tensor, vae, latents_scale, latents_bias,
                resolution: int):
    """
    Full augmentation: apply random horizontal flip + random resized crop to
    raw_image, then re-encode with the VAE so that the latent x is perfectly
    aligned with the augmented image seen by the teacher encoder.

    Color jitter is applied only to raw_image (teacher input) to give the
    encoder a harder task without changing the VAE latent distribution.

    Args:
        raw_image:     (B, C, H, W) uint8 tensor on GPU
        vae:           frozen AutoencoderKL, already on GPU
        latents_scale: (1,4,1,1) scale tensor
        latents_bias:  (1,4,1,1) bias tensor
        resolution:    target spatial resolution (int)

    Returns:
        raw_aug: (B, C, H, W) uint8 — augmented image for teacher
        x_aug:   (B, 4, H//8, W//8) float — fresh VAE latent
    """
    B = raw_image.shape[0]
    device = raw_image.device

    # Decide per-sample flip
    flip_mask = torch.rand(B, device=device) < 0.5

    # Random crop parameters (same crop applied to image and latent-via-re-encode)
    # torchvision.transforms.functional.get_params expects a PIL/Tensor image
    # We operate on float [0,1] for the crop, then convert back.
    img_float = raw_image.float() / 255.0   # (B, C, H, W) in [0, 1]

    raw_aug_list = []

    for i in range(B):
        img_i = img_float[i]   # (C, H, W)

        # Random resized crop
        top, left, height, width = T.RandomResizedCrop.get_params(
            img_i, scale=(0.8, 1.0), ratio=(3/4, 4/3)
        )
        img_i = TF.resized_crop(img_i, top, left, height, width,
                                 size=[resolution, resolution],
                                 interpolation=T.InterpolationMode.BICUBIC,
                                 antialias=True)

        # Horizontal flip
        if flip_mask[i]:
            img_i = img_i.flip(-1)

        raw_aug_list.append(img_i)

    raw_aug_float = torch.stack(raw_aug_list, dim=0)   # (B, C, H, W) float [0,1]

    # Re-encode with VAE to get aligned latents
    # VAE expects [-1, 1] input
    vae_input = raw_aug_float * 2.0 - 1.0
    posterior = vae.encode(vae_input).latent_dist
    z = posterior.sample()  # DiagonalGaussianDistribution.sample() handles reparameterisation
    x_aug = z * latents_scale + latents_bias   # apply the standard scaling

    # Color jitter only on the teacher's raw image (does not affect latent)
    brightness, contrast, saturation, hue = 0.2, 0.2, 0.2, 0.05
    raw_aug_jitter = []
    for i in range(B):
        img_i = raw_aug_float[i]
        fn_order = torch.randperm(4)
        for fn_id in fn_order:
            if fn_id == 0:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * brightness
                img_i = TF.adjust_brightness(img_i, max(0, factor))
            elif fn_id == 1:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * contrast
                img_i = TF.adjust_contrast(img_i, max(0, factor))
            elif fn_id == 2:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * saturation
                img_i = TF.adjust_saturation(img_i, max(0, factor))
            else:
                factor = (torch.rand(1).item() * 2 - 1) * hue
                img_i = TF.adjust_hue(img_i, max(-0.5, min(0.5, factor)))
        raw_aug_jitter.append(img_i)

    raw_aug_float = torch.stack(raw_aug_jitter, dim=0)
    # Convert back to uint8 for the teacher preprocessing pipeline
    raw_aug = (raw_aug_float * 255.0).clamp(0, 255).to(torch.uint8)

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
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params   = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(
        accelerator_unwrap(model).named_parameters()
    )
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# Module-level reference set in main() so update_ema can call unwrap cleanly
_accelerator = None
def accelerator_unwrap(model):
    if _accelerator is not None:
        return _accelerator.unwrap_model(model)
    # Fallback: strip DDP wrapper manually
    return getattr(model, 'module', model)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def uses_repa(args):
    if args.enc_type is None:
        return False
    return args.enc_type.lower() not in {"none", "no-repa", "baseline"} and args.proj_coeff > 0


def tracker_name(args):
    return args.report_to.lower()


def should_log(args):
    return tracker_name(args) != "none"


def uses_wandb(args):
    return tracker_name(args) == "wandb"


#################################################################################
# Training Loop                                                                 #
#################################################################################

def main(args):
    global _accelerator

    logging_dir = Path(args.output_dir, args.exp_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    log_with = None if not should_log(args) else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )
    _accelerator = accelerator  # expose for update_ema

    # Logger is only created on main process; other ranks get a no-op.
    logger = logging.getLogger(__name__)  # safe no-op fallback for non-main ranks

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    use_repa = uses_repa(args)
    if use_repa:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
    else:
        encoders, encoder_types, architectures = [], [], []

    z_dims = [encoder.embed_dim for encoder in encoders]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    # FIX: pass cfg_prob → class_dropout_prob so --cfg-prob=0.2 reaches LabelEmbedder
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
    ema   = deepcopy(model).to(device)
    vae   = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    requires_grad(vae, False)   # make intent explicit; VAE is always frozen

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)

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
        logger.info(f"CFG dropout prob (class_dropout_prob): {args.cfg_prob}")
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

    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
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
            map_location='cpu',
            weights_only=False,
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        if should_log(args):
            init_kwargs = {}
            if uses_wandb(args):
                init_kwargs["wandb"] = {"name": f"{args.exp_name}"}
            accelerator.init_trackers(
                project_name="REPA",
                config=tracker_config,
                init_kwargs=init_kwargs,
            )

    # Progress reported via plain print every --log-every steps (no tqdm)
    _log_every = getattr(args, 'log_every', 100)

    # Fixed sample batch for qualitative logging
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys  = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    n   = ys.size(0)
    xT  = torch.randn((n, 4, latent_size, latent_size), device=device)

    # FIX: initialise grad_norm before the loop so it is always defined
    grad_norm = torch.tensor(0.0, device=device)

    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            step_start_time = time.perf_counter()

            raw_image = raw_image.to(device)
            x         = x.squeeze(dim=1).to(device)
            y         = y.to(device)

            # ------------------------------------------------------------------
            # Augmentation
            # Two modes selected by --augment-mode:
            #
            #   "vae"       (default, correct): augment raw_image with random
            #               crop + flip + jitter, then re-encode with VAE so
            #               that latent x is aligned with the augmented image.
            #               Slightly heavier (one extra VAE forward pass per
            #               step) but removes the teacher/latent mismatch.
            #
            #   "flip_only" (fast fallback): only a random horizontal flip,
            #               applied identically to raw_image and the precomputed
            #               latent.  No extra VAE pass.  Suitable when GPU
            #               memory or time is tight (e.g. Colab free tier).
            #               Pass --augment-mode=flip_only to activate.
            # ------------------------------------------------------------------
            if args.augment:
                if args.augment_mode == "vae":
                    with torch.no_grad():
                        raw_image, x = augment_vae(
                            raw_image, vae, latents_scale, latents_bias,
                            args.resolution
                        )
                else:  # flip_only
                    raw_image, x = augment_flip_only(raw_image, x)
                    # x from dataset is still in moment form when flip_only is used;
                    # decode to latent after flipping
                    with torch.no_grad():
                        x = sample_posterior(x, latents_scale=latents_scale,
                                             latents_bias=latents_bias)
            else:
                with torch.no_grad():
                    x = sample_posterior(x, latents_scale=latents_scale,
                                         latents_bias=latents_bias)

            # CFG label dropout is handled entirely inside LabelEmbedder
            # (class_dropout_prob=args.cfg_prob was passed to the model constructor).
            # The --legacy branch is kept for backward compatibility only.
            if args.legacy:
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels   = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y  # LabelEmbedder applies dropout internally during training

            # Teacher forward pass
            zs = []
            teacher_start_time = time.perf_counter()
            if use_repa:
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type:
                            z = z[:, 1:]
                        if 'dinov2' in encoder_type:
                            z = z['x_norm_patchtokens']
                        zs.append(z)
            teacher_time = time.perf_counter() - teacher_start_time

            # Diffusion model forward + loss
            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                denoising_loss, proj_loss, div_loss = loss_fn(
                    model, x, model_kwargs, zs=zs
                )

                denoising_loss_mean = denoising_loss.mean()
                proj_loss_mean      = proj_loss.mean()
                div_loss_mean       = div_loss.mean()

                total_loss = (
                    denoising_loss_mean
                    + proj_loss_mean * args.proj_coeff
                    + div_loss_mean  * args.div_coeff
                )

                accelerator.backward(total_loss)

                # FIX: only clip + capture grad_norm on actual optimizer steps
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # FIX: EMA update uses unwrapped model (no brittle module. stripping)
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
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Sampling / qualitative logging
            if accelerator.sync_gradients and (
                (args.sample_at_step_one and global_step == 1) or
                (global_step % args.sampling_steps == 0 and global_step > 0)
            ):
                from samplers import euler_sampler
                ema.eval()
                with torch.no_grad():
                    samples = euler_sampler(
                        ema,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale=args.sample_cfg_scale,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples    = vae.decode((samples    - latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs      - latents_bias) / latents_scale).sample
                    samples    = (samples    + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    gt_samples  = accelerator.gather(gt_samples.to(torch.float32))
                if uses_wandb(args) and wandb is not None:
                    accelerator.log({
                        "samples":    wandb.Image(array2grid(out_samples)),
                        "gt_samples": wandb.Image(array2grid(gt_samples)),
                    })
                logging.info("Generating EMA samples done.")

            # ---------------------------------------------------------------
            # Logging — all losses are raw (unweighted) for interpretability
            # ---------------------------------------------------------------
            step_time = time.perf_counter() - step_start_time

            # grad_norm: use the value from the last sync step (stale by at most
            # gradient_accumulation_steps-1 micro-steps, but always defined)
            grad_norm_val = accelerator.gather(grad_norm).mean().detach().item() \
                if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

            logs = {
                # Total weighted loss (what the optimiser actually sees)
                "loss/total":     accelerator.gather(total_loss).mean().detach().item(),
                # Raw component losses (unweighted) for diagnosing training dynamics
                "loss/denoising": accelerator.gather(denoising_loss_mean).mean().detach().item(),
                "loss/proj":      accelerator.gather(proj_loss_mean).mean().detach().item(),
                "loss/div":       accelerator.gather(div_loss_mean).mean().detach().item(),
                # Optimisation health
                "grad_norm":      grad_norm_val,
                # Timing
                "step_time":      step_time,
                "teacher_time":   teacher_time,
            }
            if accelerator.is_main_process and global_step % _log_every == 0:
                print(
                    f"[step {global_step:>6d}/{args.max_train_steps}]"
                    f"  diff={logs['loss/denoising']:.4f}"
                    f"  proj={logs['loss/proj']:.4f}"
                    f"  div={logs['loss/div']:.4f}"
                    f"  gn={grad_norm_val:.3f}"
                    f"  loss={logs['loss/total']:.4f}"
                    f"  {logs['step_time']:.2f}s/step",
                    flush=True,
                )
            if should_log(args):
                accelerator.log(logs, step=global_step)

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

    # logging
    parser.add_argument("--output-dir",  type=str, default="exps")
    parser.add_argument("--exp-name",    type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to",   type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--sample-at-step-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100, help="Print loss every N optimizer steps")
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model",        type=str)
    parser.add_argument("--num-classes",  type=int, default=1000)
    parser.add_argument("--encoder-depth",type=int, default=8)
    parser.add_argument("--fused-attn",   action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",      action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir",   type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # augmentation
    parser.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=False,
        help="Enable online augmentation."
    )
    parser.add_argument(
        "--augment-mode", type=str, default="vae",
        choices=["vae", "flip_only"],
        help=(
            "vae (default): random crop+flip+jitter then re-encode with VAE — "
            "correct but one extra VAE forward pass per step.  "
            "flip_only: random horizontal flip only, applied to both raw_image "
            "and precomputed latent — fast, no extra VAE pass, good Colab fallback."
        ),
    )

    # precision
    parser.add_argument("--allow-tf32",     action="store_true")
    parser.add_argument("--mixed-precision",type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs",                   type=int,   default=1400)
    parser.add_argument("--max-train-steps",          type=int,   default=400000)
    parser.add_argument("--checkpointing-steps",      type=int,   default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate",            type=float, default=1e-4)
    parser.add_argument("--adam-beta1",               type=float, default=0.9)
    parser.add_argument("--adam-beta2",               type=float, default=0.999)
    parser.add_argument("--adam-weight-decay",        type=float, default=0.)
    parser.add_argument("--adam-epsilon",             type=float, default=1e-08)
    parser.add_argument("--max-grad-norm",            type=float, default=1.0)

    # seed / workers
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    # loss / encoder
    parser.add_argument("--path-type",  type=str,   default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str,   default="v", choices=["v"])
    parser.add_argument("--cfg-prob",   type=float, default=0.1,
                        help="Label dropout probability — passed to LabelEmbedder "
                             "as class_dropout_prob AND used by --legacy branch.")
    parser.add_argument("--enc-type",   type=str,   default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting",  type=str,   default="uniform")
    parser.add_argument("--legacy",     action=argparse.BooleanOptionalAction, default=False)

    # diversity loss
    parser.add_argument(
        "--div-coeff", type=float, default=0.0,
        help="Weight for the batch feature variance diversity loss (0 = disabled)."
    )

    # sampling CFG
    parser.add_argument(
        "--sample-cfg-scale", type=float, default=2.0,
        help="CFG scale used for mid-training sample logging."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)