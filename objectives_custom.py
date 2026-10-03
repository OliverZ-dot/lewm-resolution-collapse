"""Two planning-time Objectives used in an earlier exploration of
model-derived auxiliary costs (see Appendix E, "Motivating Evidence", in
the paper). Both are drop-in `swm.planning.Objective`s, combined with the
standard `GoalMSE` via `swm.planning.WeightedSum` -- no changes to the
model or training loop.

Idea 1 (`GaussianNormPenalty`): SIGReg trains the projector so that
projections of `emb` along random directions look like draws from a
standard Gaussian (see `module.SIGReg`). That's a *population*-level
statistic (needs a batch to estimate the characteristic function) and
can't be evaluated on a single candidate trajectory. The per-candidate
proxy used here follows from the same target distribution: if
`emb ~ N(0, I_D)` then `||emb||^2 ~ chi2(D)` (mean D, std sqrt(2D)). A
candidate whose rolled-out embedding norm sits many std's away from D is
"implausible" under the model's own training objective, regardless of how
close it lands to the goal embedding.

Idea 2 (`LinearProbeMSE`): applies a frozen linear probe (fit by
`fit_probe.py` via ridge regression from cached real embeddings ->
physical state/proprio) to `predicted_emb`/`goal_emb`'s last step, and
costs the resulting physical-quantity-space MSE instead of (additively
with) raw embedding MSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNormPenalty(nn.Module):
    """Chi-squared-norm implausibility proxy for SIGReg's isotropic-Gaussian
    training target (see module docstring). Reads `predicted_emb` (B,S,T,D)
    from `info_dict`, skips the first `history_len` context frames (which are
    encoded, not predicted), and returns the T-averaged per-candidate penalty
    (B,S). `history_len` must match the `plan_config.history_len` used to
    build the rollout (defaults to 1, the value used by every env config in
    this repo)."""

    def __init__(self, pred_key: str = 'predicted_emb', history_len: int = 1) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.history_len = history_len

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]  # (B, S, T, D)
        future = pred_emb[:, :, self.history_len:, :]
        D = future.size(-1)
        sq_norm = future.pow(2).sum(dim=-1)  # (B, S, T-history_len)
        z = (sq_norm - D) / (2.0 * D) ** 0.5
        return z.pow(2).mean(dim=-1)  # (B, S)


class LinearProbeMSE(nn.Module):
    """MSE, in physical-quantity space, between a frozen linear probe applied
    to `predicted_emb`'s last step and the same probe applied to
    `goal_emb`'s last step. `probe` is a plain `nn.Linear(emb_dim,
    quantity_dim)` with `requires_grad_(False)` -- see `fit_probe.py`."""

    def __init__(
        self,
        probe: nn.Linear,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
    ) -> None:
        super().__init__()
        self.probe = probe
        for p in self.probe.parameters():
            p.requires_grad_(False)
        self.pred_key = pred_key
        self.goal_key = goal_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key][..., -1, :]  # (B, S, D)
        goal_emb = info_dict[self.goal_key][:, None, -1, :].expand_as(pred_emb)
        pred_q = self.probe(pred_emb)
        goal_q = self.probe(goal_emb.detach())
        return F.mse_loss(pred_q, goal_q.detach(), reduction='none').sum(dim=-1)
