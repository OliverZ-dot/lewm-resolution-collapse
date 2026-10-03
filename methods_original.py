"""Original planners that are not another L2 weight.

The certified recipe is GoalMSE + small action-L2 through the whole CEM
loop. wmean rewrites that prior and ties it. Everything else that still
talked about L2 lost. These methods use different objects:

  ProgressPath     score the whole predicted approach to the goal, not
                   only the last frame. Terminal GoalMSE is an unresolved
                   blob; a trajectory that is getting closer is a
                   situation-level question the model can still answer.
  MultiHorizon     last-step + mid-horizon GoalMSE. A plan that only
                   looks good at H is the failure mode of an unresolved
                   elite (the model hallucinates a close z_H).
  MonotonePath     penalise predicted steps that move *away* from the goal.
                   A geometric prior on latent paths, not on actions.
  ModalElite       do not average across action modes. k-means the elite;
                   the next CEM mean is the largest cluster. Bagging
                   stays *inside* a mode.
  ConsensusElite   reweight the elite by proximity to its own mean.
                   Prefer the members the committee already agreed on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_worldmodel as swm
from stable_worldmodel.planning.solver.callbacks import Callback


def _pred_goal(info_dict):
    pred = info_dict['predicted_emb']
    goal = info_dict['goal_emb'][:, None, -1:, :].expand_as(pred)
    return pred, goal


class ProgressPath(nn.Module):
    """Mean latent distance to the goal over the predicted horizon."""

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        return (pred - goal.detach()).pow(2).sum(dim=tuple(range(2, pred.ndim)))


class MultiHorizon(nn.Module):
    """Terminal GoalMSE plus the same cost at the midpoint of the horizon."""

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        t = pred.size(2)
        mid = max(0, t // 2)
        last = F.mse_loss(pred[:, :, -1:, :], goal[:, :, -1:, :].detach(), reduction='none')
        mid_t = F.mse_loss(pred[:, :, mid:mid + 1, :], goal[:, :, mid:mid + 1, :].detach(), reduction='none')
        dims = tuple(range(2, last.ndim))
        return last.sum(dim=dims) + mid_t.sum(dim=dims)


class FrameGoalMSE(nn.Module):
    """GoalMSE at a single predicted-embedding index."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        t = pred.size(2)
        i = min(max(self.index, 0), t - 1)
        sl = F.mse_loss(
            pred[:, :, i:i + 1, :],
            goal[:, :, i:i + 1, :].detach(),
            reduction='none',
        )
        return sl.sum(dim=tuple(range(2, sl.ndim)))


class DecayingPath(nn.Module):
    """GoalMSE on every predicted frame, weights falling with horizon.

    Context index 0 is dropped. Frame t (1-indexed from the first prediction)
    gets weight (T - t), so the shortest rollout is trusted most. This is the
    planner-side counterpart of the resolution-vs-horizon curve.
    """

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        dist = (pred - goal.detach()).pow(2).sum(dim=-1)
        t = dist.size(-1)
        w = torch.arange(t, 0, -1, device=dist.device, dtype=dist.dtype)
        w[0] = 0
        return (dist * w).sum(dim=-1)


class MidOnly(nn.Module):
    """GoalMSE at the midpoint only. Ablation: is the mid frame doing the work?"""

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        t = pred.size(2)
        mid = max(0, t // 2)
        mid_t = F.mse_loss(
            pred[:, :, mid:mid + 1, :],
            goal[:, :, mid:mid + 1, :].detach(),
            reduction='none',
        )
        return mid_t.sum(dim=tuple(range(2, mid_t.ndim)))


class FrameGoalMSE(nn.Module):
    """GoalMSE at a single predicted-emb index (0 is usually context)."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        t = pred.size(2)
        i = min(max(self.index, 0), t - 1)
        sl = F.mse_loss(
            pred[:, :, i:i + 1, :],
            goal[:, :, i:i + 1, :].detach(),
            reduction='none',
        )
        return sl.sum(dim=tuple(range(2, sl.ndim)))


class DecayingPath(nn.Module):
    """Weight earlier predicted frames more: w_t = (T-1-t) for t>=1, w_0=0.

    Follows the resolution diagnosis: short-horizon questions keep ordinal
    signal inside the elite, the last frame does not.
    """

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        dist = (pred - goal.detach()).pow(2).sum(dim=-1)
        t = dist.size(-1)
        w = torch.arange(t - 1, -1, -1, device=dist.device, dtype=dist.dtype)
        w[0] = 0
        return (dist * w).sum(dim=-1)


class TripleHorizon(nn.Module):
    """GoalMSE at start, mid, and end of the predicted horizon."""

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        t = pred.size(2)
        idxs = sorted({0, max(0, t // 2), t - 1})
        total = None
        for i in idxs:
            sl = F.mse_loss(
                pred[:, :, i:i + 1, :],
                goal[:, :, i:i + 1, :].detach(),
                reduction='none',
            )
            val = sl.sum(dim=tuple(range(2, sl.ndim)))
            total = val if total is None else total + val
        return total


class MonotonePath(nn.Module):
    """Sum of increases in latent distance to the goal along the prediction."""

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred, goal = _pred_goal(info_dict)
        dist = (pred - goal.detach()).pow(2).sum(dim=-1)
        if dist.size(-1) < 2:
            return dist.mean(dim=-1)
        inc = (dist[..., 1:] - dist[..., :-1]).clamp(min=0)
        return inc.mean(dim=-1)


class ModalElite(Callback):
    """Replace the uniform elite mean with the mean of the largest action mode."""

    name = 'modal_elite'

    def __init__(self, n_clusters: int = 2, n_iter: int = 8) -> None:
        super().__init__(reduction='none')
        self.n_clusters = int(n_clusters)
        self.n_iter = int(n_iter)

    def compute(self, **state):
        elites = state['topk_candidates']
        mean = state['mean']
        b, k, h, d = elites.shape
        x = elites.reshape(b, k, -1)
        c = min(self.n_clusters, k)
        # farthest-point init from the current mean
        centers = []
        remain = x
        ref = x.mean(dim=1)
        for _ in range(c):
            dist = (remain - ref.unsqueeze(1)).pow(2).sum(-1)
            idx = dist.argmax(dim=1)
            pick = remain[torch.arange(b, device=x.device), idx]
            centers.append(pick)
            ref = pick
        centers = torch.stack(centers, dim=1)
        assign = torch.zeros(b, k, dtype=torch.long, device=x.device)
        for _ in range(self.n_iter):
            dist = (x.unsqueeze(2) - centers.unsqueeze(1)).pow(2).sum(-1)
            assign = dist.argmin(dim=-1)
            for j in range(c):
                mask = (assign == j).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1)
                centers[:, j] = (x * mask).sum(dim=1) / denom
        counts = torch.stack([(assign == j).sum(dim=1) for j in range(c)], dim=1)
        best = counts.argmax(dim=1)
        new = centers[torch.arange(b, device=x.device), best].reshape(b, h, d)
        mean.copy_(new)
        return {'n_modes_used': int(c), 'largest': int(counts[0, best[0]].item())}


class ConsensusElite(Callback):
    """Reweight elites by closeness to the elite mean (committee consensus)."""

    name = 'consensus_elite'

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__(reduction='none')
        self.temperature = float(temperature)

    def compute(self, **state):
        elites = state['topk_candidates']
        mean = state['mean']
        center = elites.mean(dim=1, keepdim=True)
        dist = (elites - center).pow(2).sum(dim=(2, 3))
        scale = dist.std(dim=1, keepdim=True).clamp_min(1e-8)
        w = torch.softmax(-dist / (self.temperature * scale), dim=1)
        mean.copy_((elites * w[:, :, None, None]).sum(dim=1))
        return {'entropy': float((-(w * w.clamp_min(1e-12).log()).sum(1)).mean())}
