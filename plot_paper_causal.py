"""Figure: causal evidence that the collapse tracks CEM's own target frame,
not a fixed physical horizon (Exp 16/18). Two panels:
  (a) elite rho vs. env-steps, H=5 vs H=8, TwoRoom and Reacher
  (b) terminal-frame elite rho as a function of H = 5,6,7,8 (dose-response)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / 'iclr2027' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

ENVS = ['tworoom', 'reacher']
LABEL = {'tworoom': 'TwoRoom', 'reacher': 'Reacher'}
COLOR = {'tworoom': '#2166ac', 'reacher': '#1a9850'}
HORIZONS = [5, 6, 7, 8]


def load_h5(env):
    d = json.loads((REPO / f'outputs/runs/{env}/exp15_resolution_curve/curve.json').read_text())
    recs, frame_idxs, marks = d['records'], d['layout']['frame_idxs'], d['layout']['marks']
    out = {}
    for i in frame_idxs:
        xs = np.array([r['frames'][f't{i}']['rho_same_elite'] for r in recs], float)
        out[marks[f't{i}']] = float(np.nanmean(xs))
    return out


def load_h(env, h):
    p = REPO / f'outputs/runs/{env}/exp16_horizon_length/h{h}.json'
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    recs, frame_idxs, marks = d['records'], d['layout']['frame_idxs'], d['layout']['marks']
    out = {}
    for i in frame_idxs:
        xs = np.array([r['frames'][f't{i}']['rho_same_elite'] for r in recs], float)
        out[marks[f't{i}']] = float(np.nanmean(xs))
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # panel (a): H=5 vs H=8 curves over env-steps
    ax = axes[0]
    for env in ENVS:
        h5 = load_h5(env)
        h8 = load_h(env, 8)
        xs5 = sorted(h5)
        xs8 = sorted(h8)
        ax.plot(xs5, [h5[x] for x in xs5], marker='o', ls='--', color=COLOR[env], alpha=0.55,
                lw=1.8, ms=6, label=f'{LABEL[env]}, H=5 (terminal at 25)')
        ax.plot(xs8, [h8[x] for x in xs8], marker='s', ls='-', color=COLOR[env],
                lw=2.2, ms=6, label=f'{LABEL[env]}, H=8 (terminal at 40)')
    ax.axhline(0.0, color='#999999', lw=1.0, ls=':')
    ax.axvline(25, color='#888888', lw=1.0, ls=':')
    ax.axvline(40, color='#888888', lw=1.0, ls=':')
    ax.text(25, 0.92, 'H=5\nterminal', ha='center', fontsize=8.5, color='#555555')
    ax.text(40, 0.92, 'H=8\nterminal', ha='center', fontsize=8.5, color='#555555')
    ax.set_xlabel('env-steps into the rollout', fontsize=11)
    ax.set_ylabel(r'elite Spearman $\rho$', fontsize=11)
    ax.set_title('(a) Stretching the horizon moves the collapse point\nwith it, at the same env-step count', fontsize=11.5)
    ax.set_ylim(-0.15, 1.0)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.92)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)

    # panel (b): terminal-frame rho vs H, dose-response
    ax = axes[1]
    for env in ENVS:
        vals = []
        for h in HORIZONS:
            curve = load_h5(env) if h == 5 else load_h(env, h)
            vals.append(curve[max(curve)])
        ax.plot(HORIZONS, vals, marker='o', color=COLOR[env], lw=2.4, ms=9, label=LABEL[env])
    ax.axhline(0.0, color='#999999', lw=1.0, ls=':')
    ax.axhspan(-0.1, 0.1, color='#cccccc', alpha=0.35, zorder=0)
    ax.set_xlabel('CEM planning horizon $H$', fontsize=11)
    ax.set_ylabel(r'elite Spearman $\rho$ at the terminal frame', fontsize=11)
    ax.set_title('(b) The terminal frame stays at chance level\nat every horizon tried', fontsize=11.5)
    ax.set_xticks(HORIZONS)
    ax.set_ylim(-0.15, 0.5)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    out = OUT / 'causal_horizon.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
