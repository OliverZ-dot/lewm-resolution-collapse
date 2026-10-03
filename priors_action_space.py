"""Model-free action-space priors, for the reality check in Direction 3.

Direction 1 shows the world model cannot order the candidates inside its own
elite set. If that is right, then whatever is producing the gains reported for
auxiliary costs in latent-space MPC cannot be coming from model-derived
information about those candidates -- and a prior defined purely on the action
sequence, which needs no model at all, should do at least as well.

This module supplies the model-free arms for that comparison:

  ActionSmoothness   a cost on how jerky the action sequence is
  colored_noise_cem  an iCEM-style sampling prior (Pinneri et al., 2020) that
                     biases the *proposal* toward temporally correlated
                     sequences instead of adding a cost term

Both operate on the env-step action axis rather than the solver's horizon axis.
A candidate is stored as ``(H, action_block * env_action_dim)``; the sequence the
robot actually executes is that tensor viewed as ``(H * action_block,
env_action_dim)``, and temporal structure only means anything on the latter.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

import stable_worldmodel.planning.solver.cem as _cem_mod


class ActionSmoothness(nn.Module):
    """Penalizes squared successive differences of the executed action sequence.

    Reads ``action_candidates`` of shape ``(B, S, H, action_block * D)``, views it
    as ``(B, S, H * action_block, D)`` and returns ``sum_t ||a_{t+1} - a_t||^2``
    per candidate, shape ``(B, S)``.
    """

    def __init__(self, env_action_dim: int, action_key: str = 'action_candidates') -> None:
        super().__init__()
        self.env_action_dim = int(env_action_dim)
        self.action_key = action_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        a = info_dict[self.action_key]
        b, s = a.shape[0], a.shape[1]
        seq = a.reshape(b, s, -1, self.env_action_dim)
        d = seq[:, :, 1:, :] - seq[:, :, :-1, :]
        return d.pow(2).sum(dim=(2, 3))


def _colored_noise(shape, beta: float, time_len: int, env_action_dim: int,
                   generator: torch.Generator, device, dtype, randn) -> torch.Tensor:
    """White noise shaped to a 1/f^beta power spectrum along the env-step axis.

    `shape` is the solver's requested ``(B, S, H, action_block * D)``. We build the
    noise on ``(B, S, H * action_block, D)``, where the time axis is contiguous in
    execution order, then view it back. beta=0 recovers plain white noise.

    `randn` must be the *unpatched* ``torch.randn``: this function runs while the
    shim below is installed, and its own draw is 4-D, so calling the patched name
    would recurse.
    """
    b, s, h, ad = shape
    n = time_len
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    nz = freqs > 0
    scale[nz] = freqs[nz] ** (-beta / 2.0)
    # The f=0 term needs a finite scale, not zero. Zeroing it removes the
    # constant-offset direction from the proposal, so no candidate can hold a
    # sustained action over the horizon -- which is most of what these tasks
    # require, and which collapsed success rate by up to 53 points when we tried
    # it. The standard treatment is a low-frequency cutoff, so f=0 inherits the
    # scale of the lowest resolvable frequency.
    scale[0] = scale[1] if len(scale) > 1 else 1.0
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)

    n_freq = freqs.shape[0]
    re = randn(b, s, n_freq, env_action_dim, generator=generator, device=device)
    im = randn(b, s, n_freq, env_action_dim, generator=generator, device=device)
    spec = torch.complex(re, im) * scale_t.view(1, 1, -1, 1)
    sig = torch.fft.irfft(spec, n=n, dim=2)

    # unit variance per (batch, sample, dim) so beta does not change the step size
    sig = sig / (sig.std(dim=2, keepdim=True) + 1e-8)
    return sig.reshape(b, s, h, ad).to(dtype)


@contextmanager
def colored_noise_cem(beta: float, num_samples: int, horizon: int,
                      action_dim: int, env_action_dim: int):
    """Make `CEMSolver.solve` draw temporally correlated candidates.

    `cem.py` has exactly one ``torch.randn`` call, the candidate draw, so that is
    what we intercept. Patching the attribute reaches the global ``torch``
    module, so the shim is guarded on the *exact* candidate shape
    ``(bs, num_samples, horizon, action_dim)`` and forwards anything else to the
    real ``torch.randn``. Callers should assert the hit count is what they expect.
    """
    if beta == 0.0:
        yield lambda: 0
        return

    real_randn = torch.randn
    time_len = horizon * (action_dim // env_action_dim)
    want = (num_samples, horizon, action_dim)
    hits = [0]

    def shim(*size, **kw):
        if len(size) == 4 and tuple(size[1:]) == want:
            hits[0] += 1
            return _colored_noise(
                size, beta, time_len, env_action_dim,
                kw.get('generator'), kw.get('device'),
                kw.get('dtype', torch.float32), real_randn,
            )
        return real_randn(*size, **kw)

    _cem_mod.torch.randn = shim
    try:
        yield lambda: hits[0]
    finally:
        _cem_mod.torch.randn = real_randn
