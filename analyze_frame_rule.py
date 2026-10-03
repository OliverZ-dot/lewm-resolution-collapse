"""Exp 19: a principled, calibration-driven frame-selection rule, instead of
sweeping frame index against success rate.

Exp 15b picked the "best" frame per environment by looking at *success rate*,
which is the target metric itself -- five frame indices times four
environments is twenty configurations, and reporting the best one per
environment invites the obvious objection: this is a fitted hyperparameter,
not a principled method, and it may not transfer to a new environment.

This experiment asks: can the frame be chosen from a signal that has nothing
to do with the planning outcome -- specifically Exp 15a's elite-Spearman
resolution curve, which is a property of the *model's own predictions*,
measured by re-executing candidates and comparing predicted vs. true cost --
and applied with **one threshold, fixed in advance and shared across all four
environments**?

Rule (v1, absolute threshold): frame* = max{i : rho(i) >= tau}, fallback to
i=1 if nothing clears tau. This was tried first and **failed** on TwoRoom:
its curve never exceeds 0.5 (peak 0.493 at t3), so tau=0.5 falls through to
the frame-1 fallback -- and Exp 15b already showed frame_1 is *actively
harmful* on TwoRoom (-16.5pp vs. last, the worst arm in the whole sweep).
An absolute threshold conflates "how correlated this environment's latents
are in general" with "has resolution collapsed *for this environment*", and
the fallback direction (shortest = safest) is empirically backwards. Keeping
this failure in the writeup because it is informative: it is why v2 uses a
threshold relative to each environment's own short-horizon resolution rather
than one absolute number across environments.

Rule (v2, relative threshold, used below): choose

    frame* = max{ i : rho(i) >= frac * rho(1) }

i.e. the latest frame that still retains at least a `frac` fraction of the
resolution the model has at the shortest possible horizon. This adapts to
each environment's own baseline correlation level instead of assuming 0.5
means the same thing everywhere, and it never degenerates to the
frame-1 fallback in practice (frame 1 trivially satisfies its own criterion,
so the rule only moves *forward* from there as far as resolution allows).

Because Exp 15a and Exp 15b already share the same checkpoints, eval pairs,
and certified-L2 methodology, frame*'s success rate does not need a new
planner run -- it is already sitting in Exp 15b's `sweep.json` as one of the
`frame_k` arms. This script only does the (blind, pre-registered-style)
selection and the lookup, plus a sensitivity sweep over `frac` so the
conclusion does not hinge on one cherry-picked threshold.

    python analyze_frame_rule.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ENVS = ['tworoom', 'reacher', 'pusht', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'reacher': 'Reacher', 'pusht': 'PushT', 'cube': 'Cube'}
FRACS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
PRIMARY_FRAC = 0.7


def load_curve(env):
    d = json.loads(Path(f'outputs/runs/{env}/exp15_resolution_curve/curve.json').read_text())
    recs = d['records']
    frame_idxs = d['layout']['frame_idxs']
    rho = {}
    for i in frame_idxs:
        xs = np.array([r['frames'][f't{i}']['rho_same_elite'] for r in recs], float)
        rho[i] = float(np.nanmean(xs))
    return rho, max(frame_idxs)


def load_sweep(env):
    return json.loads(Path(f'outputs/runs/{env}/exp15_frame_sweep/sweep.json').read_text())


def choose_frame(rho: dict, horizon: int, frac: float) -> int:
    """frame* = latest frame retaining >= frac of the shortest-horizon rho."""
    tau = frac * rho[min(rho)]
    ok = [i for i in sorted(rho) if rho[i] >= tau]
    if len(ok) == horizon:
        return horizon
    return max(ok) if ok else 1


def mcnemar_p(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int(np.sum((~a) & b))
    n10 = int(np.sum(a & (~b)))
    n = n01 + n10
    if n == 0:
        return 1.0
    return binomtest(min(n01, n10), n, 0.5, alternative='two-sided').pvalue


def main():
    print('Calibration curve (Exp 15a elite rho by frame, n=50, blind to success rate):')
    curves, horizons = {}, {}
    for env in ENVS:
        rho, h = load_curve(env)
        curves[env], horizons[env] = rho, h
        print(f'  {LABEL[env]:9s} ' + '  '.join(f't{i}={rho[i]:+.3f}' for i in sorted(rho)))
    print()

    print('Frame chosen by rule (frame* = max i with rho(i) >= frac * rho(1)), frac sweep:')
    print(f"{'env':9s} " + ' '.join(f'frac={t:.2f}' for t in FRACS))
    for env in ENVS:
        row = [choose_frame(curves[env], horizons[env], t) for t in FRACS]
        print(f'{LABEL[env]:9s} ' + ' '.join(f'{v:^9d}' for v in row))
    print()

    print(f'=== Primary rule, frac={PRIMARY_FRAC} ===')
    print(f"{'env':9s} {'frame*':>6s} {'last':>8s} {'rule SR':>8s} {'oracle-best':>12s} "
          f"{'rule vs last':>14s} {'rule vs oracle':>15s}")
    print('-' * 90)
    for env in ENVS:
        sw = load_sweep(env)
        arms = sw['arms']
        frame_star = choose_frame(curves[env], horizons[env], PRIMARY_FRAC)
        key = 'last' if frame_star == horizons[env] else f'frame_{frame_star}'
        rule = arms[key]
        last = arms['last']
        frame_arms = {k: v for k, v in arms.items() if k.startswith('frame_')}
        oracle_key = max(frame_arms, key=lambda k: frame_arms[k]['sr'])
        oracle = frame_arms[oracle_key]

        p_vs_last = mcnemar_p(rule['success'], last['success'])
        p_vs_oracle = mcnemar_p(rule['success'], oracle['success'])
        star1 = '*' if p_vs_last < 0.05 else ''
        star2 = '*' if p_vs_oracle < 0.05 else ''

        oracle_str = f'{oracle_key}({oracle["sr"]:.1f}%)'
        print(f"{LABEL[env]:9s} {key:>6s} {last['sr']:7.1f}% {rule['sr']:7.1f}% "
              f"{oracle_str:>12s} "
              f"{rule['sr'] - last['sr']:+6.1f}pp p={p_vs_last:.3f}{star1:1s} "
              f"{rule['sr'] - oracle['sr']:+6.1f}pp p={p_vs_oracle:.3f}{star2:1s}")
    print()
    print('rule vs last   : does the *blindly chosen* frame beat the terminal-frame baseline?')
    print('rule vs oracle : how much of the success-rate-sweep\'s best-case gain does the blind rule recover?')


if __name__ == '__main__':
    main()
