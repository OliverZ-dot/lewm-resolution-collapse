"""Auxiliary planning costs and matched random-cost control utilities.

Appendix E of the paper shows that a substantial share of the success-rate
gain from adding a model-derived auxiliary cost to a CEM cost is reproduced
by magnitude-matched noise. That result is about one cost. To claim it is a
property of *auxiliary costs on sampling-based planners* rather than a quirk
of one specific term, the same control has to be run against costs we did
not invent. This module supplies two:

* `RNDNovelty` -- Random Network Distillation (Burda et al. 2019), the standard
  novelty bonus. Fixed random target network, predictor trained to match it on
  in-distribution latents, novelty = squared prediction error. Cited in our own
  introduction as an alternative to implausibility, so it is the fairest
  possible external comparison.
* `swm.planning.ControlPenalty` needs no wrapper -- it already ships with the
  framework and appears in essentially every MPC paper.

Both need a *matched* random control, and matching is the subtle part. Adding a
constant to every candidate does nothing to a ranking-based optimizer, so the
quantity that has to be matched is the per-call spread across candidates, not
the mean. `SpreadRecorder` measures it and `MatchedRandomCost` reproduces it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RNDNovelty(nn.Module):
    """Random Network Distillation novelty on predicted latents.

    Reads ``predicted_emb`` ``(B, S, T, D)``, skips the first ``history_len``
    context frames (encoded, not predicted), and returns the T-averaged squared
    error between a trained predictor and a frozen random target ``(B, S)``.

    Both networks are small MLPs over a single latent, so the per-candidate cost
    is two matmuls on top of a rollout the planner already paid for. This is the
    same accounting as `GaussianNormPenalty`, which keeps the comparison honest:
    neither cost gets to win by being given more compute.

    The predictor is fit by `train_rnd.py` on latents encoded from the real
    dataset, so novelty is low on states the model has seen and high elsewhere.
    """

    def __init__(
        self,
        target: nn.Module,
        predictor: nn.Module,
        pred_key: str = 'predicted_emb',
        history_len: int = 1,
    ) -> None:
        super().__init__()
        self.target = target
        self.predictor = predictor
        for p in self.target.parameters():
            p.requires_grad_(False)
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.predictor.eval()
        self.pred_key = pred_key
        self.history_len = history_len

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]  # (B, S, T, D)
        future = pred_emb[:, :, self.history_len :, :]
        flat = future.reshape(-1, future.size(-1))
        err = (self.predictor(flat) - self.target(flat)).pow(2).mean(dim=-1)
        return err.view(future.shape[:3]).mean(dim=-1)  # (B, S)


def make_rnd_nets(emb_dim: int, hidden: int = 256, out_dim: int = 64, seed: int = 0):
    """Build the (target, predictor) pair. The target is random and frozen
    forever; only the predictor is trained. Seeded so the target is identical
    across the baseline, real-cost and control arms of an experiment."""
    gen = torch.Generator().manual_seed(seed)

    def mlp(depth: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = emb_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # orthogonal-ish init drawn from our own generator so the two
                # nets are reproducible independently of global torch state
                w = torch.empty_like(m.weight)
                nn.init.normal_(w, std=1.0 / math.sqrt(m.in_features))
                with torch.no_grad():
                    m.weight.copy_(w)
                    m.bias.zero_()
        return net

    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    target = mlp(depth=1)
    predictor = mlp(depth=2)  # deeper, as in the original: it has to fit
    return target, predictor


class SpreadRecorder(nn.Module):
    """Transparent wrapper that records the per-call standard deviation of a
    cost across candidates.

    What a magnitude-matched control has to match is this number. An additive
    term that is constant across candidates leaves every ranking unchanged and
    therefore cannot affect what the planner does, so matching the *mean* of a
    cost would be matching the wrong thing. Matching the spread is what makes
    "same magnitude" mean "same ability to reorder candidates".
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.spreads: list[float] = []

    def reset(self) -> None:
        self.spreads = []

    def forward(self, info_dict: dict) -> torch.Tensor:
        cost = self.inner(info_dict)
        if cost.ndim == 2 and cost.size(1) > 1:
            self.spreads.append(float(cost.std(dim=1).mean()))
        return cost

    def summary(self) -> float:
        return float(torch.tensor(self.spreads).median()) if self.spreads else 0.0


class MatchedRandomCost(nn.Module):
    """Information-free cost whose spread across candidates matches a target.

    Emits ``scale * chi2_1`` noise, the same distributional shape as
    `RandomCostPenalty` used in the paper's original control, but with `scale`
    set so the across-candidate standard deviation equals `target_spread`. A
    chi-squared variable with one degree of freedom has standard deviation
    ``sqrt(2)``, hence the divisor.

    The values depend only on a fixed seed, never on the candidate, so any
    effect this cost has on success rate is an effect of perturbing the
    optimizer rather than of measuring anything.
    """

    def __init__(self, target_spread: float, seed: int = 0) -> None:
        super().__init__()
        self.target_spread = float(target_spread)
        self.scale = self.target_spread / math.sqrt(2.0)
        self.gen = torch.Generator().manual_seed(seed)

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict['predicted_emb']  # (B, S, T, D)
        shape = pred_emb.shape[:2]
        noise = torch.randn(shape, generator=self.gen).to(pred_emb.device)
        return self.scale * noise.pow(2)
