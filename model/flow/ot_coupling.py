import warnings
from functools import partial

import numpy as np
import ot as pot
import torch


class OTPlanSampler:
    """Minibatch OT coupling for OT-CFM (Tong et al. 2024; Pooladian et al. 2023).

    Mirrors the default behavior of torchcfm.OTPlanSampler:
      - exact EMD via POT (`pot.emd`)
      - squared L2 cost on flattened tensors
      - uniform marginals
      - no cost normalization
      - `sample_plan` draws (i, j) ~ pi with replacement (the standard recipe
        described in both papers' algorithm sections; the deterministic
        Hungarian variant is not used by either paper or torchcfm's default).
    """

    def __init__(self, method="exact", reg=0.05, num_threads=1, warn=True):
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        else:
            raise ValueError(f"Unknown OT method: {method}")
        self.warn = warn

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)) or np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan; reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan_with_labels(self, x0, x1, y1=None, replace=True):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j], (y1[j] if y1 is not None else None)
