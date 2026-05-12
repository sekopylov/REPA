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
        div_coeff=0.0,          # NEW: weight for diversity loss (0 = disabled)
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

        For each projector head, compute the per-dimension std of the
        projected features across the batch and penalize low variance with:
            loss = mean(max(0, gamma - std(z_d)))
        where gamma=1 is the target minimum std.

        This encourages the model to spread its representations across the batch,
        countering the memorization collapse where all outputs map to nearly
        identical features.

        Returns a scalar tensor (already averaged over heads and dimensions).
        """
        device = zs_tilde[0].device if zs_tilde else torch.device('cpu')
        if not zs_tilde or self.div_coeff == 0.0:
            return torch.tensor(0.0, device=device)

        gamma = 1.0
        total = torch.tensor(0.0, device=device)
        count = 0
        for z_tilde in zs_tilde:
            # z_tilde: (B, T, D) — T patch tokens, D feature dim
            B, T, D = z_tilde.shape
            z_flat = z_tilde.reshape(B * T, D)                      # (B*T, D)
            z_flat = F.normalize(z_flat, dim=-1)                     # L2-normalize before variance
            std = z_flat.std(dim=0)                                  # (D,) per-dim std over batch
            hinge = F.relu(gamma - std)                              # (D,) penalize dims below gamma
            total = total + hinge.mean()
            count += 1

        return total / count if count > 0 else torch.tensor(0.0, device=device)

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

        # NEW: diversity loss — scalar, broadcast to match denoising_loss shape for gather compatibility
        div_loss_val = self.diversity_loss(zs_tilde)
        div_loss = torch.full_like(denoising_loss, div_loss_val.item())

        return denoising_loss, proj_loss, div_loss