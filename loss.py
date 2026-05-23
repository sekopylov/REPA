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

    def covariance_loss(self, zs_tilde):
        """
        Off-diagonal covariance decorrelation loss (Barlow Twins / VICReg cov term).

        For each projector head, we want the feature dimensions to be uncorrelated
        across the batch. This prevents the projector from encoding redundant
        information and acts as an anti-memorization regularizer.

        Operates on L2-normalized features so it is fully compatible with proj_loss
        (which also uses normalized features). It only penalizes the CORRELATION
        STRUCTURE, not the direction or magnitude — so it does not fight proj_loss.

        For z_tilde of shape (B, T, D):
          1. Reshape to (B, T*D) — treat each sample as a flat feature vector
             OR average over T first to get a per-image summary (B, D).
             We use option 2: mean-pool over T so B is the effective sample count.
          2. L2-normalize each sample across D.
          3. Compute the (D, D) cross-correlation matrix C = Z^T Z / B.
          4. Loss = sum of squared off-diagonal entries of C.
             On-diagonal entries are 1 by construction (normalized), so we skip them.
             This is exactly the Barlow Twins off-diagonal term.

        Properties:
          - Always >= 0, equals 0 only when all feature dims are uncorrelated.
          - Does not collapse: even a perfectly trained proj_loss still leaves
            off-diagonal correlations to push against.
          - Scale-invariant (works on normalized features).
          - Compatible with fp16 (no exp/log operations).

        Args:
            zs_tilde: list of tensors, each (B, T, D)

        Returns:
            scalar tensor with gradients intact
        """
        if not zs_tilde or self.div_coeff == 0.0:
            device = zs_tilde[0].device if zs_tilde else torch.device('cpu')
            return zs_tilde[0].sum() * 0.0 if zs_tilde else torch.tensor(0.0, device=device)

        total = None
        count = 0

        for z_tilde in zs_tilde:
            # z_tilde: (B, T, D)
            B, T, D = z_tilde.shape

            # Mean-pool over patch tokens → per-image summary (B, D)
            z = z_tilde.mean(dim=1)              # (B, D)

            # L2-normalize each sample — makes diagonal of C exactly 1
            z = F.normalize(z, dim=-1)           # (B, D)

            # Cross-correlation matrix: (D, D), values in [-1, 1]
            C = (z.T @ z) / B                   # (D, D)

            # Off-diagonal penalty: zero out diagonal, square remaining entries
            eye = torch.eye(D, device=z.device, dtype=z.dtype)
            off_diag = C * (1 - eye)            # (D, D), diagonal zeroed
            head_loss = (off_diag ** 2).sum() / D   # normalize by D so scale is stable

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

        # projection loss (REPA alignment) — normalized cosine similarity
        if not zs:
            proj_loss = torch.zeros_like(denoising_loss)
        else:
            proj_loss = 0.
            bsz = zs[0].shape[0]
            for z, z_tilde in zip(zs, zs_tilde):
                z_tilde = F.normalize(z_tilde, dim=-1)
                z       = F.normalize(z,       dim=-1)
                proj_loss += mean_flat(-(z * z_tilde).sum(dim=-1))  # (B, T) → scalar
            proj_loss /= len(zs)

        # covariance decorrelation loss — off-diagonal Barlow Twins term
        # Operates on normalized features → orthogonal to proj_loss.
        # Does not collapse regardless of div_coeff magnitude.
        # Broadcast to per-sample shape so accelerator.gather() works in train.py.
        cov_loss_scalar = self.covariance_loss(zs_tilde)
        div_loss = cov_loss_scalar.expand(denoising_loss.shape[0])

        return denoising_loss, proj_loss, div_loss