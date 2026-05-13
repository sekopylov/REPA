import torch
import numpy as np
import torch.nn.functional as F


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x):
    """Take the sum over all non-batch dimensions."""
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SILoss:
    def __init__(
        self,
        prediction='v',
        path_type="linear",
        weighting="uniform",
        encoders=[],
        accelerator=None,
        latents_scale=None,
        latents_bias=None,
        div_coeff=0.0,
    ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.div_coeff = div_coeff

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def diversity_loss(self, zs_tilde):
        """
        Batch feature variance loss (VICReg-style variance term).

        Computes std across the BATCH dimension only (not across patches).
        Rationale: reshaping (B, T, D) → (B*T, D) and computing std over
        B*T=16384 samples causes std >> gamma=1.0 immediately, so the hinge
        F.relu(gamma - std) is always 0 after step ~50 and no gradient flows.

        Instead we average per-patch std over T, each computed over B samples.
        With B=32-64, std per dim hovers around 0.3-0.8 for typical projector
        outputs, making gamma=0.5 a sustained, reachable target.

        Returns a scalar tensor that stays in the computation graph.
        """
        if not zs_tilde or self.div_coeff == 0.0:
            device = zs_tilde[0].device if zs_tilde else torch.device('cpu')
            return zs_tilde[0].sum() * 0.0 if zs_tilde else torch.tensor(0.0, device=device)

        gamma = 0.5   # reachable target for per-dim std over batch of 32-64
        total = None
        count = 0
        for z_tilde in zs_tilde:
            # z_tilde: (B, T, D)
            # Compute std over B for each of the T patch positions, then average over T
            # std shape: (T, D) — mean over T gives scalar per head
            std = z_tilde.std(dim=0, unbiased=False)    # (T, D)
            hinge = F.relu(gamma - std)                 # (T, D)
            head_loss = hinge.mean()                    # scalar, gradients intact
            total = head_loss if total is None else total + head_loss
            count += 1

        return total / count

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs is None:
            model_kwargs = {}

        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError()

        model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss (REPA alignment)
        if not zs:
            proj_loss = torch.zeros_like(denoising_loss)
        else:
            proj_loss = 0.
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = F.normalize(z_tilde_j, dim=-1)
                    z_j = F.normalize(z_j, dim=-1)
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)

        # diversity loss — scalar tensor, gradients intact (NOT detached via .item())
        # Broadcast to a per-sample vector matching denoising_loss shape so that
        # accelerator.gather() works correctly in the training loop.
        div_loss_scalar = self.diversity_loss(zs_tilde)
        div_loss = div_loss_scalar.expand(denoising_loss.shape[0])

        return denoising_loss, proj_loss, div_loss