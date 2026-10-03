"""Exp 17 summary: does MultiHorizon (mid+last+L2) beat a spread-matched
random-cost control, or is its gain a tie-breaking confound like the one
found for implausibility (Exp 5)?

    python analyze_multi_h_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ENVS = ['tworoom', 'reacher', 'pusht', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'reacher': 'Reacher', 'pusht': 'PushT', 'cube': 'Cube'}


def mcnemar_p(a, b):
    """Exact one-sided McNemar (sign test on discordant pairs), a vs b."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int(np.sum((~a) & b))   # b succeeds, a fails
    n10 = int(np.sum(a & (~b)))   # a succeeds, b fails
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    p = binomtest(min(n01, n10), n, 0.5, alternative='two-sided').pvalue
    return p, n01, n10


def main():
    print(f"{'env':9s} {'last':>8s} {'multi_h_l2':>12s} {'randmid_l2':>12s} "
          f"{'mh vs last':>12s} {'mh vs rand':>12s} {'rand vs last':>13s}")
    print('-' * 90)
    for env in ENVS:
        p = Path(f'outputs/runs/{env}/exp17_multi_h_control/control.json')
        if not p.exists():
            print(f'{LABEL[env]:9s}  (no data yet)')
            continue
        d = json.loads(p.read_text())
        arms = d['arms']
        if not all(k in arms for k in ('last', 'multi_h_l2', 'randmid_l2')):
            have = list(arms.keys())
            print(f'{LABEL[env]:9s}  partial: have {have}')
            continue
        last = arms['last']['success']
        mh = arms['multi_h_l2']['success']
        rand = arms['randmid_l2']['success']
        sr_last, sr_mh, sr_rand = arms['last']['sr'], arms['multi_h_l2']['sr'], arms['randmid_l2']['sr']

        p_mh_last, n01, n10 = mcnemar_p(mh, last)
        p_mh_rand, n01b, n10b = mcnemar_p(mh, rand)
        p_rand_last, _, _ = mcnemar_p(rand, last)

        def fmt(pv, win):
            star = '*' if pv < 0.05 else ('.' if pv < 0.1 else ' ')
            return f'{win:+.1f}pp p={pv:.3f}{star}'

        print(f"{LABEL[env]:9s} {sr_last:7.1f}% {sr_mh:11.1f}% {sr_rand:11.1f}%  "
              f"{fmt(p_mh_last, sr_mh - sr_last):>12s} {fmt(p_mh_rand, sr_mh - sr_rand):>12s} "
              f"{fmt(p_rand_last, sr_rand - sr_last):>13s}")
        print(f"           spreads: last={d['spreads']['last']:.3g} mid={d['spreads']['mid']:.3g} "
              f"mh={d['spreads']['mh']:.3g} l2={d['spreads']['l2']:.3g}  lam_l2={d['lam_l2']:.3g}")
    print()
    print('mh vs last   : does adding the mid-horizon term help at all?')
    print('mh vs rand   : does the mid-horizon term beat noise of the same spread? <- the control')
    print('rand vs last : does *any* noise of that spread help (the confound Exp5 found for implausibility)?')


if __name__ == '__main__':
    main()
