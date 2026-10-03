"""Analysis for Experiment 8: does widening the elite set remove the confound?

Reads `outputs/runs/<env>/exp8_elite_width/elite_width.json`.

The prediction from Section 11 is that the gain a spread-matched random cost can
buy should shrink as the elite fraction k/N grows, because k controls how hard CEM
commits to an ordering the model cannot resolve. At k = N the refit stops depending
on the ordering at all.

Two things have to be checked before a decaying curve counts as evidence.

Floor effects. A gain also decays trivially if large k simply destroys
performance, since there is nothing left to gain. So the baseline success rate is
plotted alongside, and any k whose baseline has collapsed is marked rather than
silently averaged in.

Tuning parity. Every k received the same full weight sweep, and each is credited
with its best setting. Had we reused the strength selected at the default k = 30,
that k would hold a tuning advantage and the decay would be partly manufactured.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import binomtest, spearmanr  # noqa: E402

ENVS = ['tworoom', 'pusht', 'reacher', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'pusht': 'PushT', 'reacher': 'Reacher', 'cube': 'Cube'}
FIG_DIR = Path('figures/direction1')
COLLAPSE_FRAC = 0.6  # baseline below this share of its best value counts as collapsed


def mcnemar(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    return (binomtest(n01, n, 0.5).pvalue if n else 1.0)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    per_env, rows = {}, []

    for env in ENVS:
        p = Path(f'outputs/runs/{env}/exp8_elite_width/elite_width.json')
        if not p.exists():
            print(f'[{env}] missing, skipped')
            continue
        d = json.load(open(p))
        ns = d['num_samples']
        ks, fracs, bases, gains, ps, best_r = [], [], [], [], [], []
        for k_str, e in sorted(d['by_topk'].items(), key=lambda kv: int(kv[0])):
            if 'baseline_sr' not in e or not e.get('random'):
                continue
            g = {r: v['sr'] - e['baseline_sr'] for r, v in e['random'].items()}
            br = max(g, key=g.get)
            ks.append(int(k_str))
            fracs.append(int(k_str) / ns)
            bases.append(e['baseline_sr'])
            gains.append(g[br])
            best_r.append(br)
            ps.append(mcnemar(e['baseline_success'], e['random'][br]['success']))
        if len(ks) < 2:
            print(f'[{env}] only {len(ks)} elite widths finished, skipped')
            continue
        bases_a = np.array(bases)
        collapsed = bases_a < COLLAPSE_FRAC * bases_a.max()
        per_env[env] = dict(ks=np.array(ks), fracs=np.array(fracs), bases=bases_a,
                            gains=np.array(gains), ps=np.array(ps),
                            best_r=best_r, collapsed=collapsed, ns=ns)

        ok = ~collapsed
        rho = spearmanr(np.array(ks)[ok], np.array(gains)[ok]).statistic if ok.sum() > 2 else np.nan
        rows.append({'env': env, 'rho_k_vs_gain': rho,
                     'gain_at_min_k': gains[0], 'gain_at_max_k': gains[-1],
                     'k_min': ks[0], 'k_max': ks[-1],
                     'n_collapsed': int(collapsed.sum())})

    if not per_env:
        print('no data yet')
        return

    for env, d in per_env.items():
        print(f'\n[{LABEL[env]}] elite width sweep (num_samples={d["ns"]})')
        print(f"  {'topk':>5s} {'frac':>6s} {'baseline':>9s} {'best random gain':>17s} "
              f"{'at r':>7s} {'p':>9s}")
        for i in range(len(d['ks'])):
            flag = '  <- baseline collapsed' if d['collapsed'][i] else ''
            print(f"  {d['ks'][i]:5d} {d['fracs'][i]:6.3f} {d['bases'][i]:8.1f}% "
                  f"{d['gains'][i]:+17.1f} {d['best_r'][i]:>7s} "
                  f"{d['ps'][i]:9.1e}{flag}")

    if rows:
        print(f"\n{'env':9s} {'gain at smallest k':>19s} {'gain at largest k':>18s} "
              f"{'rho(k, gain)':>13s} {'collapsed k':>12s}")
        for r in rows:
            print(f"{LABEL[r['env']]:9s} {r['gain_at_min_k']:+14.1f} (k={r['k_min']:<3d}) "
                  f"{r['gain_at_max_k']:+13.1f} (k={r['k_max']:<3d}) "
                  f"{r['rho_k_vs_gain']:+13.3f} {r['n_collapsed']:12d}")
        print('\nA negative rho(k, gain) is the predicted direction: the wider the '
              'elite set, the\nless a random cost can buy. Rows with collapsed '
              'baselines are excluded from rho,\nsince a gain cannot shrink '
              'meaningfully once there is no performance left to gain.')

    # ---- figure ---------------------------------------------------------
    envs = list(per_env)
    fig, axes = plt.subplots(1, len(envs), figsize=(4.0 * len(envs), 4.6), squeeze=False)
    for ax, env in zip(axes[0], envs):
        d = per_env[env]
        ax.axhline(0, color='k', lw=0.9)
        ax.plot(d['fracs'], d['gains'], 'o-', color='#C44E52', ms=6,
                label='best random-cost gain')
        for i in range(len(d['ks'])):
            if d['ps'][i] < 0.05:
                ax.plot(d['fracs'][i], d['gains'][i], '*', color='k', ms=11, zorder=6)
            if d['collapsed'][i]:
                ax.plot(d['fracs'][i], d['gains'][i], 'x', color='0.2', ms=13,
                        mew=2, zorder=7)
        ax.set_xscale('log')
        ax.set_xlabel('elite fraction  k / N')
        ax.set_ylabel('success rate gain (points)')
        ax.set_title(LABEL[env], fontsize=11)

        ax2 = ax.twinx()
        ax2.plot(d['fracs'], d['bases'], 's--', color='#4C72B0', ms=4, lw=1.1,
                 label='baseline')
        ax2.set_ylabel('baseline success rate (%)', color='#4C72B0')
        ax2.tick_params(axis='y', labelcolor='#4C72B0')
        ax2.set_ylim(0, 100)
    axes[0][0].text(0.02, 0.03, 'star: p<0.05 vs baseline\ncross: baseline collapsed',
                    transform=axes[0][0].transAxes, fontsize=7.5, color='0.3',
                    va='bottom')
    fig.suptitle('What a random cost can buy, as the planner is made to commit '
                 'less hard to an ordering it cannot resolve', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / 'elite_width.png'
    fig.savefig(out, dpi=160)
    print(f'\nsaved -> {out}')

    with open('outputs/runs/direction1_elite_width_summary.json', 'w') as f:
        json.dump(rows, f, indent=2, default=float)


if __name__ == '__main__':
    main()
