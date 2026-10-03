"""Curiosity-JEPA: intrinsic-motivation exploration on top of LeWM.

Additive module — `jepa.py` / `module.py` are never modified. This file
provides:

  - `EnsembleDynamics`: wraps a trained/training `JEPA` (jepa.py) with K
    independent predictor heads (same architecture as the paper's
    `ARPredictor`) sharing the base encoder, and exposes the
    `stable_worldmodel.protocols.Dynamics` surface (`encode`/`rollout`)
    the planning stack needs.
  - `RunningProjectionDensity`: reuses SIGReg's own random-projection
    machinery (see `module.SIGReg`) to keep a running per-direction
    histogram of visited embeddings, and scores how "unlikely" a new
    embedding is under that empirical distribution (novelty).
  - `IntrinsicObjective`: a `stable_worldmodel.protocols.Objective` that
    turns ensemble disagreement + projection novelty, accumulated over an
    imagined rollout, into a per-candidate CEM cost (more surprising =
    lower cost, since CEM minimizes).

Note: this module is not exercised by the diagnostic experiments in the
paper (Sections 4-5); it is imported transitively via `explore_train.py`,
which also supports a curiosity-driven exploration policy used in earlier,
unrelated exploration of the codebase.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from jepa import JEPA, detach_clone
from module import ARPredictor


class EnsembleDynamics(nn.Module):
    """Adds K independent `ARPredictor` heads to a base `JEPA` model.

    The base model's own `predictor` keeps being trained/used as usual
    (e.g. for the standard LeWM eval / GoalMSE planning); the ensemble
    heads are additional, separately-initialized predictors used only to
    estimate epistemic disagreement for curiosity-driven exploration.

    Args:
        jepa: The base `JEPA` model (encoder + predictor + action_encoder
            + projector/pred_proj), exactly as built from
            `config/train/model/lewm.yaml`.
        num_heads: Number of ensemble predictor heads (K).
        predictor_kwargs: Kwargs used to build each `ARPredictor` head.
            Must match the base predictor's architecture for a fair
            comparison (num_frames, depth, heads, mlp_dim, input_dim,
            hidden_dim, output_dim, dim_head, dropout, emb_dropout).
        bootstrap_prob: Per-sample keep-probability used when computing
            each head's training loss (Bernoulli bootstrap masking,
            Plan2Explore/Osband-style), so heads see different effective
            data subsets despite sharing one dataloader batch.
    """

    def __init__(
        self,
        jepa: JEPA,
        num_heads: int,
        predictor_kwargs: dict,
        bootstrap_prob: float = 0.8,
        history_size: int = 3,
    ):
        super().__init__()
        self.jepa = jepa
        self.num_heads = num_heads
        self.bootstrap_prob = bootstrap_prob
        self.history_size = history_size
        self.ensemble_predictors = nn.ModuleList(
            [ARPredictor(**predictor_kwargs) for _ in range(num_heads)]
        )

    # -- Dynamics protocol -------------------------------------------------

    def encode(self, info: dict) -> dict:
        return self.jepa.encode(info)

    def predict_heads(
        self, emb: torch.Tensor, act_emb: torch.Tensor
    ) -> torch.Tensor:
        """Run all K heads on the same (emb, act_emb) context.

        Args:
            emb: (N, T, D) shared context embedding.
            act_emb: (N, T, A) shared context action embedding.

        Returns:
            (K, N, T, D) per-head predictions (post pred_proj, matching
            `JEPA.predict`'s output convention).
        """
        outs = []
        for head in self.ensemble_predictors:
            pred = head(emb, act_emb)
            pred = self.jepa.pred_proj(
                rearrange(pred, 'n t d -> (n t) d')
            )
            pred = rearrange(pred, '(n t) d -> n t d', n=emb.size(0))
            outs.append(pred)
        return torch.stack(outs, dim=0)

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        """Roll out K ensemble heads (in addition to the base predictor).

        Mirrors `JEPA.rollout`'s bookkeeping but also advances an ensemble
        of trajectories in parallel, using the *per-head* prediction to
        advance *that head's own* copy of the trajectory (so disagreement
        compounds correctly over the horizon instead of collapsing back to
        a shared mean at every step).

        Populates, on top of everything `JEPA.rollout` populates:
          - `predicted_emb`: (B, S, H+horizon, D) consensus (mean-of-heads)
            trajectory — kept so goal-distance objectives / visualization
            code paths that expect this key keep working unmodified.
          - `ensemble_emb`: (K, B, S, H+horizon, D) every head's own
            trajectory.
        """
        assert 'pixels' in info_dict, 'pixels not in info_dict'
        H = info_dict['pixels'].size(2)
        B, S, T = action_candidates.shape[:3]
        act_0, act_future = torch.split(action_candidates, [H, T - H], dim=2)
        info_dict['action'] = act_0
        n_steps = T - H

        _init = {
            k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
        }
        _init = self.jepa.encode(_init)
        base_emb = _init['emb'].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        base_emb = rearrange(base_emb, 'b s ... -> (b s) ...').clone()
        act = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future = rearrange(act_future, 'b s ... -> (b s) ...')

        HS = self.history_size

        # one growing trajectory per ensemble head, all starting identical
        K = self.num_heads
        head_embs = [base_emb.clone() for _ in range(K)]
        act_per_head = [act.clone() for _ in range(K)]

        for t in range(n_steps):
            next_act = act_future[:, t : t + 1, :]
            preds_this_step = []
            for k, head in enumerate(self.ensemble_predictors):
                emb_k, act_k = head_embs[k], act_per_head[k]
                act_emb_k = self.jepa.action_encoder(act_k)
                emb_trunc = emb_k[:, -HS:]
                act_trunc = act_emb_k[:, -HS:]
                pred = head(emb_trunc, act_trunc)[:, -1:]
                pred = self.jepa.pred_proj(pred.squeeze(1)).unsqueeze(1)
                preds_this_step.append(pred)
                head_embs[k] = torch.cat([emb_k, pred], dim=1)
                act_per_head[k] = torch.cat([act_k, next_act], dim=1)

        # final step (predict the last state for every head, no action to append)
        for k, head in enumerate(self.ensemble_predictors):
            emb_k, act_k = head_embs[k], act_per_head[k]
            act_emb_k = self.jepa.action_encoder(act_k)
            emb_trunc = emb_k[:, -HS:]
            act_trunc = act_emb_k[:, -HS:]
            pred = head(emb_trunc, act_trunc)[:, -1:]
            pred = self.jepa.pred_proj(pred.squeeze(1)).unsqueeze(1)
            head_embs[k] = torch.cat([emb_k, pred], dim=1)

        ensemble_traj = torch.stack(head_embs, dim=0)  # (K, BS, H+horizon, D)
        ensemble_traj = rearrange(
            ensemble_traj, 'k (b s) ... -> k b s ...', b=B, s=S
        )
        consensus = ensemble_traj.mean(dim=0)  # (B, S, H+horizon, D)

        info_dict['predicted_emb'] = consensus
        info_dict['ensemble_emb'] = ensemble_traj
        info_dict['emb'] = _init['emb'].unsqueeze(1).expand(B, S, -1, -1)
        return info_dict

    # -- training-time losses (not part of the Dynamics protocol) ---------

    def ensemble_loss(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Bootstrap-masked MSE loss for the K ensemble heads.

        Args:
            emb: (B, T, D) ground-truth context embeddings for one training
                clip (T = history_size + num_preds), already encoded and
                detached from the base model's own graph (`sg(...)`, as in
                the paper's `pred_loss`).
            act_emb: (B, T, A) corresponding action embeddings.

        Returns:
            Scalar loss, averaged over heads (and, within each head, over
            only the bootstrap-kept samples in the batch).
        """
        B = emb.size(0)
        target = emb[:, 1:].detach()
        total = 0.0
        for head in self.ensemble_predictors:
            pred = head(emb[:, :-1], act_emb[:, :-1])
            pred = self.jepa.pred_proj(
                rearrange(pred, 'b t d -> (b t) d')
            )
            pred = rearrange(pred, '(b t) d -> b t d', b=B)
            err = (pred - target).pow(2).mean(dim=(1, 2))  # (B,)
            keep = torch.bernoulli(
                torch.full_like(err, self.bootstrap_prob)
            )
            denom = keep.sum().clamp_min(1.0)
            total = total + (err * keep).sum() / denom
        return total / self.num_heads


class RunningProjectionDensity(nn.Module):
    """Free-riding novelty scorer built on SIGReg's own random projections.

    SIGReg (module.py) tests whether a *batch* of embeddings looks Gaussian
    along `num_proj` random directions. Here we keep one fixed set of
    directions and, per direction, a running histogram of projected values
    seen so far in the replay buffer. A new point's novelty is the average
    (over directions) negative log-density under that empirical histogram:
    a point whose projections all fall in densely-visited bins looks
    "boring"; a point whose projections fall in sparse/empty bins looks
    "novel" — exactly the same directions SIGReg already uses to keep the
    *training* distribution close to isotropic-Gaussian.

    This is intentionally simple (fixed-width histograms, not full KDE);
    treat the bin count / range as things to ablate, per the design doc.
    """

    def __init__(
        self,
        dim: int,
        num_proj: int = 1024,
        num_bins: int = 64,
        value_range: float = 6.0,
        seed: int = 0,
    ):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        A = torch.randn(dim, num_proj, generator=gen)
        A = A / A.norm(p=2, dim=0)
        self.register_buffer('A', A)  # (D, M)
        self.num_bins = num_bins
        self.value_range = value_range
        # +2 for open-ended overflow bins on both sides
        self.register_buffer(
            'counts', torch.ones(num_proj, num_bins + 2)
        )  # Laplace-smoothed (start at 1, not 0)
        self.register_buffer('total', torch.tensor(float(num_bins + 2)))

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """Add a batch of embeddings `z` (N, D) to the running histograms."""
        proj = z @ self.A  # (N, M)
        idx = self._bin_index(proj)  # (N, M) in [0, num_bins+1]
        M = proj.size(1)
        col = torch.arange(M, device=z.device).unsqueeze(0).expand_as(idx)
        flat_idx = (col * self.counts.size(1) + idx).reshape(-1)
        self.counts.view(-1).index_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=self.counts.dtype)
        )
        self.total += proj.size(0)

    def _bin_index(self, proj: torch.Tensor) -> torch.Tensor:
        scaled = (proj + self.value_range) / (2 * self.value_range) * self.num_bins
        idx = scaled.floor().long() + 1  # shift by 1 for the low-overflow bin
        return idx.clamp(0, self.num_bins + 1)

    @torch.no_grad()
    def novelty(self, z: torch.Tensor) -> torch.Tensor:
        """Average negative log-density novelty score for `z` (..., D)."""
        shape = z.shape[:-1]
        flat = z.reshape(-1, z.shape[-1])
        proj = flat @ self.A  # (N, M)
        idx = self._bin_index(proj)  # (N, M)
        M = proj.size(1)
        col = torch.arange(M, device=z.device).unsqueeze(0).expand_as(idx)
        counts_at = self.counts[col, idx]  # (N, M)
        density = counts_at / self.total.clamp_min(1.0)
        nll = -torch.log(density.clamp_min(1e-12))
        score = nll.mean(dim=-1)  # (N,)
        return score.reshape(shape)


class IntrinsicObjective(nn.Module):
    """`stable_worldmodel.protocols.Objective`: disagreement + novelty cost.

    Reads `ensemble_emb` (K, B, S, H+horizon, D) and `predicted_emb`
    (B, S, H+horizon, D) from an already-rolled-out `info_dict` (populated
    by `EnsembleDynamics.rollout`) and returns a per-candidate CEM cost of
    shape (B, S) — the *negative* of the accumulated intrinsic reward,
    since CEM minimizes cost.

    Args:
        density: A `RunningProjectionDensity` instance (kept up to date by
            the training loop, read-only here).
        w_disagreement: Weight on the ensemble-disagreement term.
        w_novelty: Weight on the projection-novelty term.
        context_len: Number of leading context frames to exclude from the
            reward sum (only the *imagined future* should be "surprising";
            the context is ground truth, not something to explore into).
    """

    def __init__(
        self,
        density: RunningProjectionDensity,
        w_disagreement: float = 1.0,
        w_novelty: float = 1.0,
        context_len: int = 3,
    ):
        super().__init__()
        self.density = density
        self.w_disagreement = w_disagreement
        self.w_novelty = w_novelty
        self.context_len = context_len

    def forward(self, info_dict: dict) -> torch.Tensor:
        ensemble = info_dict['ensemble_emb']  # (K, B, S, T, D)
        future = ensemble[:, :, :, self.context_len :]  # drop ground-truth context
        disagreement = future.var(dim=0).mean(dim=-1)  # (B, S, T')
        disagreement = disagreement.sum(dim=-1)  # (B, S)

        consensus_future = info_dict['predicted_emb'][:, :, self.context_len :]
        novelty = self.density.novelty(consensus_future)  # (B, S, T')
        novelty = novelty.sum(dim=-1)  # (B, S)

        reward = self.w_disagreement * disagreement + self.w_novelty * novelty
        return -reward

    # convenience for logging / ablation without re-running the rollout
    def decompose(self, info_dict: dict) -> dict:
        ensemble = info_dict['ensemble_emb']
        future = ensemble[:, :, :, self.context_len :]
        disagreement = future.var(dim=0).mean(dim=-1).sum(dim=-1)
        consensus_future = info_dict['predicted_emb'][:, :, self.context_len :]
        novelty = self.density.novelty(consensus_future).sum(dim=-1)
        return {'disagreement': disagreement, 'novelty': novelty}
