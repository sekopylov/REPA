"""
train_sigma.py — REPA-Σ: per-batch t-conditional gradient surgery for REPA.

Forked from train.py. The only change vs the vanilla loop is the gradient
combination step. Everything else (model build, dataset, EMA, sampling, logging,
checkpointing) is preserved.

What REPA-Σ does
================
Standard REPA combines two losses with a fixed weight:

    L_total = L_diff + λ · L_repa

When the two gradients conflict (cos(g_diff, g_repa) < 0), each step partially
undoes the work of the other. Our `reports/better_grad_geometry/` measurement
shows this conflict zone exists at low-to-mid diffusion timesteps and persists
into the late training stages on CelebA SiT-B/2 (cos ≈ −0.083 at 150k, t=0.2).

REPA-Σ removes the directionally-conflicting component of g_repa before the
update — PCGrad surgery (Yu et al. 2020) applied to REPA's auxiliary loss,
restricted to REPA's parameter support (the params actually touched by L_repa:
embedders + blocks [0, encoder_depth) + projectors).

When sigma_mode == "off", this script reproduces vanilla REPA exactly (single
backward of L_diff + λ · L_repa, no surgery, no extra DDP overhead). Use that
mode as a self-consistency check.

CLI additions
=============
    --sigma-mode {off, hard, threshold, bloop}
        off:        vanilla REPA — single backward, no surgery
        hard:       project when dot(g_d_support, g_r_support) < 0
        threshold:  project when cos(g_d_support, g_r_support) < --sigma-threshold
        bloop:      use EMA of g_d as projection direction (Bloop, Apple 2024)
    --sigma-threshold FLOAT
        Cosine threshold for "threshold" mode. Default 0.0.
    --sigma-bloop-beta FLOAT
        EMA decay for Bloop mode. Default 0.99.
    --sigma-log-every INT
        Write surgery_stats.csv every N steps. Default 1 (every step).
"""
import argparse
import copy
import csv
import logging
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import torch.distributed as dist

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
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


# ─── Image preprocessing (unchanged from train.py) ────────────────────────────
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
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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


def tracker_name(args):
    return args.report_to.lower()


def should_log(args):
    return tracker_name(args) != "none"


def uses_wandb(args):
    return tracker_name(args) == "wandb"


# ─── REPA support mask ────────────────────────────────────────────────────────
def build_repa_support_set(model, encoder_depth):
    """
    Return the set of parameter names that REPA's loss can reach.

    Includes: x_embedder, t_embedder, y_embedder, blocks [0, encoder_depth).
    Excludes: blocks [encoder_depth, ...], final_layer, projectors.

    NB: projector head weights ARE touched by REPA, but we deliberately exclude
    them from the surgery scope. The projector's gradient is L_repa-only by
    construction (L_diff has zero gradient on the projector since the projector
    is fed only into the alignment loss path), so projecting it against g_d is
    undefined (g_d ≈ 0 on projector). Including projectors would create
    numerical instability without changing the diffusion update.

    This mirrors the "blocks_0_3" scope used in
    `scripts/probe_gradient_geometry_better.py`.
    """
    pre_projector_blocks = {f"blocks.{i}." for i in range(encoder_depth)}
    support = set()
    for name, _ in model.named_parameters():
        if name.startswith("projectors."):
            continue
        if (
            name.startswith("x_embedder.")
            or name.startswith("t_embedder.")
            or name.startswith("y_embedder.")
            or any(name.startswith(p) for p in pre_projector_blocks)
        ):
            support.add(name)
    return support


def strip_ddp_prefix(name):
    """`module.x_embedder.weight` → `x_embedder.weight` (when DDP-wrapped)."""
    return name[len("module."):] if name.startswith("module.") else name


# ─── Surgery primitives ───────────────────────────────────────────────────────
def compute_surgery_stats(model, support_set, eps=1e-30):
    """
    Read the current `.grad` of every parameter and compute:
      dot     = Σ_{p ∈ S} ⟨g_d[p], g_r[p]⟩   (NB: caller stored g_d AND g_r into
                                              the same .grad, this is only used
                                              with g_d.grad already loaded;
                                              prefer the explicit dict-based
                                              `surgery_inner_products` below.)
    Not used directly — kept for documentation of the math.
    """
    raise NotImplementedError("Use surgery_inner_products + apply_surgery")


def surgery_inner_products(g_d, g_r, support_set):
    """
    Compute restricted inner products in fp32:
        dot     = Σ_{p ∈ S} ⟨g_d[p], g_r[p]⟩
        norm_sq_d = Σ_{p ∈ S} ‖g_d[p]‖²
        norm_sq_r = Σ_{p ∈ S} ‖g_r[p]‖²

    g_d, g_r are dicts {stripped_name → tensor or None}. None is treated as 0.
    Sums are fp32 to avoid overflow when accumulating many fp16² values.
    """
    dot = 0.0
    norm_sq_d = 0.0
    norm_sq_r = 0.0
    for name in support_set:
        gd = g_d.get(name, None)
        gr = g_r.get(name, None)
        if gd is None or gr is None:
            continue
        gd32 = gd.detach().to(torch.float32)
        gr32 = gr.detach().to(torch.float32)
        dot += float((gd32 * gr32).sum())
        norm_sq_d += float((gd32 * gd32).sum())
        norm_sq_r += float((gr32 * gr32).sum())
    return dot, norm_sq_d, norm_sq_r


def apply_surgery_inplace(g_r, g_d, support_set, alpha):
    """
    g_r[p] ← g_r[p] - alpha · g_d[p]   for p ∈ S.
    Done in-place on the cloned g_r tensors (callers own them).

    α is a Python float (negative when removing conflict). We add (-α) · g_d
    component-wise. This is the PCGrad projection orthogonal to g_d.
    """
    for name in support_set:
        gd = g_d.get(name, None)
        gr = g_r.get(name, None)
        if gd is None or gr is None:
            continue
        gr.sub_(gd.to(gr.dtype), alpha=float(alpha))


# ─── Surgery telemetry CSV writer ─────────────────────────────────────────────
SURGERY_CSV_COLS = [
    "global_step", "step_time", "loss_diff", "loss_repa",
    "t_mean", "t_min", "t_max",
    "dot", "norm_sq_d", "norm_sq_r",
    "cos", "alpha", "projected",
    "g_d_norm", "g_r_norm",
]


def open_surgery_csv(path):
    """Open csv writer, write header if file is new, return (file, csv_writer)."""
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=SURGERY_CSV_COLS)
    if is_new:
        w.writeheader()
        f.flush()
    return f, w


# ─── Training Loop ────────────────────────────────────────────────────────────
def main(args):
    logging_dir = Path(args.output_dir, args.exp_name, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    log_with = None if not should_log(args) else args.report_to

    # For the surgery branch we sidestep DDP's auto-reducer entirely by using
    # torch.autograd.grad() and manually all-reducing the gradients. This avoids
    # DDP's "undefined gradient" error when L_repa back-props through only the
    # support params (embedders + blocks [0, encoder_depth)), leaving blocks
    # ≥ encoder_depth and final_layer with no autograd path to L_repa.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        with open(os.path.join(save_dir, "args.json"), 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        logger.info(f"sigma_mode={args.sigma_mode}, threshold={args.sigma_threshold}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    assert args.resolution % 8 == 0
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
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )
    model = model.to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.] * 4).view(1, 4, 1, 1).to(device)

    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )

    # Build the REPA support set from the UNWRAPPED model — names will be
    # stripped of "module." before lookup at runtime.
    repa_support = build_repa_support_set(model, args.encoder_depth)
    if accelerator.is_main_process:
        logger.info(f"REPA support: {len(repa_support)} params "
                    f"(of {sum(1 for _ in model.named_parameters())} total)")
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    # Resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + '.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu', weights_only=False,
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
                project_name="REPA-Sigma", config=tracker_config, init_kwargs=init_kwargs
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Sampling labels (unchanged from train.py)
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias)
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device).to(device)
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)

    # Open surgery telemetry CSV
    surgery_csv_path = os.path.join(args.output_dir, args.exp_name, "surgery_stats.csv")
    surgery_f, surgery_w = None, None
    if accelerator.is_main_process and args.sigma_mode != "off":
        surgery_f, surgery_w = open_surgery_csv(surgery_csv_path)

    # Bloop-mode running EMA of g_d (kept on CPU to avoid GPU mem balloon)
    g_d_ema_cpu = None  # dict {name: cpu tensor} when sigma_mode == "bloop"

    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            step_start_time = time.perf_counter()
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                teacher_start_time = time.perf_counter()
                if use_repa:
                    with accelerator.autocast():
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z[:, 1:]
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            zs.append(z)
                teacher_time = time.perf_counter() - teacher_start_time

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()

                # Capture stats that the surgery branch can use (t values are
                # sampled inside loss_fn; for telemetry we recompute mean only
                # when needed — not exposed by SILoss, so we read None).
                # For now: log loss values only. t-distribution telemetry would
                # require a small SILoss refactor; not done in v1.

                # ── Surgery decision tree ──────────────────────────────────────
                if args.sigma_mode == "off" or not use_repa:
                    # ===== Vanilla REPA path — byte-identical to train.py =====
                    total = loss_mean + proj_loss_mean * args.proj_coeff
                    accelerator.backward(total)

                    if accelerator.sync_gradients:
                        params_to_clip = model.parameters()
                        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        update_ema(ema, model)

                    sigma_telemetry = None
                else:
                    # ===== REPA-Σ path: autograd.grad + manual all-reduce ====
                    # We bypass DDP's auto-reducer because L_repa back-props
                    # only through embedders + blocks [0, encoder_depth) +
                    # projector, leaving the remaining blocks with no autograd
                    # path. DDP's reducer treats those as "undefined gradient"
                    # and errors out. torch.autograd.grad() sidesteps the
                    # reducer entirely; we all-reduce manually after surgery.

                    # 1. Collect param refs in deterministic order. autograd.grad
                    #    requires every tensor in `inputs` to have requires_grad=True,
                    #    so we filter; the dropped ones are pos_embed-style buffers
                    #    and never receive gradient anyway.
                    named_params = [
                        (n, p) for n, p in model.named_parameters() if p.requires_grad
                    ]
                    params_for_grad = [p for _, p in named_params]
                    stripped_names = [strip_ddp_prefix(n) for n, _ in named_params]

                    # 2. Scale losses (mimic GradScaler) so the gradients we
                    #    compute are in the same scaled space the existing
                    #    GradScaler bookkeeping expects.
                    scaler = accelerator.scaler  # may be None in bf16/no-amp
                    if scaler is not None:
                        sd_loss = scaler.scale(loss_mean)
                        sr_loss = scaler.scale(proj_loss_mean * args.proj_coeff)
                    else:
                        sd_loss = loss_mean
                        sr_loss = proj_loss_mean * args.proj_coeff

                    # 3. Compute both gradients via autograd.grad (no DDP).
                    #    retain_graph=True on the first call; second call frees.
                    g_diff_tup = torch.autograd.grad(
                        sd_loss, params_for_grad,
                        retain_graph=True, allow_unused=True, create_graph=False,
                    )
                    g_repa_tup = torch.autograd.grad(
                        sr_loss, params_for_grad,
                        retain_graph=False, allow_unused=True, create_graph=False,
                    )

                    # 4. Manually all-reduce both gradient sets across ranks.
                    if accelerator.num_processes > 1 and dist.is_initialized():
                        for g in g_diff_tup:
                            if g is not None:
                                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                                g.div_(accelerator.num_processes)
                        for g in g_repa_tup:
                            if g is not None:
                                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                                g.div_(accelerator.num_processes)

                    # 5. Build name → tensor dicts.
                    g_diff = {n: g for n, g in zip(stripped_names, g_diff_tup)}
                    g_repa = {n: g for n, g in zip(stripped_names, g_repa_tup)}

                    # 6. Choose projection direction (raw g_diff or Bloop EMA).
                    if args.sigma_mode == "bloop":
                        if g_d_ema_cpu is None:
                            g_d_ema_cpu = {
                                n: (v.detach().to('cpu', torch.float32).clone() if v is not None else None)
                                for n, v in g_diff.items()
                            }
                        else:
                            for n, v in g_diff.items():
                                if v is None:
                                    continue
                                v_cpu32 = v.detach().to('cpu', torch.float32)
                                if g_d_ema_cpu[n] is None:
                                    g_d_ema_cpu[n] = v_cpu32.clone()
                                else:
                                    g_d_ema_cpu[n].mul_(args.sigma_bloop_beta).add_(
                                        v_cpu32, alpha=1 - args.sigma_bloop_beta
                                    )
                        g_d_for_projection = {
                            n: (v.to(device).to(g_diff[n].dtype) if v is not None and g_diff.get(n) is not None else None)
                            for n, v in g_d_ema_cpu.items()
                        }
                    else:
                        g_d_for_projection = g_diff

                    # 7. Restricted inner products over REPA's support.
                    dot, norm_sq_d, norm_sq_r = surgery_inner_products(
                        g_d_for_projection, g_repa, repa_support
                    )
                    cos_metric = (
                        dot / (math.sqrt(norm_sq_d + 1e-30) * math.sqrt(norm_sq_r + 1e-30))
                        if norm_sq_d > 0 and norm_sq_r > 0
                        else 0.0
                    )

                    # 8. Decide projection trigger.
                    if args.sigma_mode == "hard":
                        do_project = dot < 0.0
                    elif args.sigma_mode == "threshold":
                        do_project = cos_metric < args.sigma_threshold
                    elif args.sigma_mode == "bloop":
                        do_project = dot < 0.0
                    else:
                        raise ValueError(f"Unknown sigma_mode={args.sigma_mode}")

                    if do_project and norm_sq_d > 0:
                        alpha = dot / (norm_sq_d + 1e-30)
                        apply_surgery_inplace(g_repa, g_d_for_projection, repa_support, alpha)
                        projected = True
                    else:
                        alpha = 0.0
                        projected = False

                    # 9. Assemble combined gradient into .grad.
                    #    Note grads are in *scaled* space; AcceleratedOptimizer's
                    #    step() unscales via GradScaler.unscale_ inside its hook.
                    #    Iterate over named_params (the requires_grad subset) —
                    #    params with requires_grad=False stay with .grad=None as
                    #    they had before the surgery branch.
                    for name, p in named_params:
                        sname = strip_ddp_prefix(name)
                        gd = g_diff.get(sname, None)
                        gr = g_repa.get(sname, None)
                        if gd is None and gr is None:
                            p.grad = None
                            continue
                        if gd is None:
                            p.grad = gr
                        elif gr is None:
                            p.grad = gd
                        else:
                            p.grad = gd + gr

                    # 10. Standard step. clip_grad_norm_ handles GradScaler
                    #     unscaling internally. With ga_steps=1 sync_gradients
                    #     is always True inside the accumulate() context.
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema, model)

                    sigma_telemetry = dict(
                        dot=dot, norm_sq_d=norm_sq_d, norm_sq_r=norm_sq_r,
                        cos=cos_metric, alpha=alpha, projected=projected,
                    )

            # End of accumulate scope
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if ((args.sample_at_step_one and global_step == 1) or
                (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, xT, ys, num_steps=50, cfg_scale=4.0,
                        guidance_low=0., guidance_high=1.,
                        path_type=args.path_type, heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples - latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                if uses_wandb(args) and wandb is not None:
                    accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                     "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            step_time = time.perf_counter() - step_start_time
            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "step_time": step_time,
                "teacher_time": teacher_time,
            }
            if sigma_telemetry is not None:
                logs.update({
                    "sigma_cos": sigma_telemetry["cos"],
                    "sigma_alpha": sigma_telemetry["alpha"],
                    "sigma_projected": 1.0 if sigma_telemetry["projected"] else 0.0,
                })

            progress_bar.set_postfix(**logs)
            if should_log(args):
                accelerator.log(logs, step=global_step)

            # Surgery CSV
            if (accelerator.is_main_process and sigma_telemetry is not None
                and surgery_w is not None
                and (global_step % max(1, args.sigma_log_every) == 0)):
                surgery_w.writerow({
                    "global_step": global_step,
                    "step_time": step_time,
                    "loss_diff": logs["loss"],
                    "loss_repa": logs["proj_loss"],
                    "t_mean": float("nan"),  # not exposed by SILoss; placeholder
                    "t_min": float("nan"),
                    "t_max": float("nan"),
                    "dot": sigma_telemetry["dot"],
                    "norm_sq_d": sigma_telemetry["norm_sq_d"],
                    "norm_sq_r": sigma_telemetry["norm_sq_r"],
                    "cos": sigma_telemetry["cos"],
                    "alpha": sigma_telemetry["alpha"],
                    "projected": int(sigma_telemetry["projected"]),
                    "g_d_norm": math.sqrt(sigma_telemetry["norm_sq_d"]),
                    "g_r_norm": math.sqrt(sigma_telemetry["norm_sq_r"]),
                })
                surgery_f.flush()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
        if surgery_f is not None:
            surgery_f.close()
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training (REPA-Σ)")

    # logging
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--sample-at-step-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    # seed + cpu
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    # ── REPA-Σ surgery flags ─────────────────────────────────────────────────
    parser.add_argument("--sigma-mode", type=str, default="off",
                        choices=["off", "hard", "threshold", "bloop"],
                        help="off=vanilla REPA; hard=project when dot<0; "
                             "threshold=project when cos<sigma-threshold; "
                             "bloop=EMA-stabilized g_d projection direction")
    parser.add_argument("--sigma-threshold", type=float, default=0.0,
                        help="Cosine threshold (only used when sigma-mode=threshold)")
    parser.add_argument("--sigma-bloop-beta", type=float, default=0.99,
                        help="EMA decay for Bloop-mode g_d direction estimate")
    parser.add_argument("--sigma-log-every", type=int, default=1,
                        help="Write a row to surgery_stats.csv every N global steps")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
