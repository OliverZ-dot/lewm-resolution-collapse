"""Paired McNemar of each Exp 15b arm against last+L2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ENVS = ['tworoom', 'reacher', 'pusht', 'cube']
LABEL = {
    'tworoom': 'TwoRoom', 'reacher': 'Reacher',
    'pusht': 'PushT', 'cube': 'Cube',
}


def mcnemar_p(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    return float(binomtest(n01, n, 0.5).pvalue), n01, n10


def main():
    print(f"{'env':9s} {'arm':12s} {'SR':>6s} {'Δ last':>8s} {'p':>8s}")
    print('-' * 48)
    for env in ENVS:
        p = Path(f'outputs/runs/{env}/exp15_frame_sweep/sweep.json')
        if not p.exists():
            print(f'{LABEL[env]:9s} not finished')
            continue
        d = json.loads(p.read_text())
        arms = d.get('arms') or {}
        last = arms.get('last')
        if last is None:
            print(f'{LABEL[env]:9s} no last arm yet')
            continue
        last_s = last['success']
        order = [k for k in arms if k.startswith('frame_')] + [
            k for k in ('decaying', 'multi_h', 'last') if k in arms
        ]
        for name in order:
            row = arms[name]
            pval, _, _ = mcnemar_p(last_s, row['success'])
            delta = row['sr'] - last['sr']
            star = '*' if pval < 0.05 and name != 'last' else ' '
            print(
                f"{LABEL[env]:9s} {name:12s} {row['sr']:5.1f}% "
                f"{delta:+7.1f}  {pval:8.3g}{star}"
            )
        print()


if __name__ == '__main__':
    main()
