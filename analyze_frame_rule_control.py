"""Exp 20 summary: does the Exp 19 rule-selected frame's success rate survive
a spread-matched random-cost control, or would any cost of the same
magnitude (combined with the certified L2 prior) do just as well?

    python analyze_frame_rule_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ENVS = ['tworoom', 'reacher', 'pusht', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'reacher': 'Reacher', 'pusht': 'PushT', 'cube': 'Cube'}


def mcnemar_p(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int(np.sum((~a) & b))
    n10 = int(np.sum(a & (~b)))
    n = n01 + n10
    if n == 0:
        return 1.0
    return binomtest(min(n01, n10), n, 0.5, alternative='two-sided').pvalue


def main():
    print(f"{'env':9s} {'frame*':>7s} {'last':>7s} {'frame_star':>11s} {'rand(matched)':>14s} "
          f"{'frame_star vs rand':>19s}")
    print('-' * 80)
    for env in ENVS:
        p = Path(f'outputs/runs/{env}/exp20_frame_rule_control/control.json')
        if not p.exists():
            print(f'{LABEL[env]:9s}  (no data yet)')
            continue
        d = json.loads(p.read_text())
        arms = d['arms']
        last, fs, rand = arms['last'], arms['frame_star'], arms['rand_frame_star']
        pv = mcnemar_p(fs['success'], rand['success'])
        star = '*' if pv < 0.05 else ''
        print(f"{LABEL[env]:9s} frame_{d['frame_star']:<2d} {last['sr']:6.1f}% {fs['sr']:10.1f}% "
              f"{rand['sr']:13.1f}%  {fs['sr'] - rand['sr']:+6.1f}pp p={pv:.2e}{star}")


if __name__ == '__main__':
    main()
