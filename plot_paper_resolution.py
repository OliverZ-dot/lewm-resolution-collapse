"""Figure: elite-set resolution collapses toward the terminal frame, on all
four environments (Exp 15a). This is the paper's central diagnostic plot.
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

ENVS = ['tworoom', 'reacher', 'pusht', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'reacher': 'Reacher', 'pusht': 'PushT', 'cube': 'Cube'}
COLOR = {'tworoom': '#2166ac', 'reacher': '#1a9850', 'pusht': '#e08214', 'cube': '#d73027'}
MARK = {'tworoom': 'o', 'reacher': 's', 'pusht': '^', 'cube': 'D'}


def load_curve(env):
    d = json.loads((REPO / f'outputs/runs/{env}/exp15_resolution_curve/curve.json').read_text())
    recs = d['records']
    frame_idxs = d['layout']['frame_idxs']
    marks = d['layout']['marks']
    steps, rho = [], []
    for i in frame_idxs:
        xs = np.array([r['frames'][f't{i}']['rho_same_elite'] for r in recs], float)
        steps.append(marks[f't{i}'])
        rho.append(float(np.nanmean(xs)))
    return np.array(steps), np.array(rho)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.6))
    for env in ENVS:
        steps, rho = load_curve(env)
        ax.plot(steps, rho, marker=MARK[env], color=COLOR[env], lw=2.2, ms=7,
                label=LABEL[env], zorder=3)
    ax.axhline(0.0, color='#999999', lw=1.0, ls=':', zorder=1)
    ax.axhspan(-0.08, 0.08, color='#cccccc', alpha=0.35, zorder=0, label='chance level')

    # annotate the terminal frame
    ax.annotate('terminal frame\n(what CEM ranks on)', xy=(25, 0.06), xytext=(15.5, 0.34),
                fontsize=9.5, color='#333333',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2))

    ax.set_xlabel('predicted frame (env-steps into the rollout)', fontsize=11)
    ax.set_ylabel(r'elite Spearman $\rho$(predicted cost, true cost)', fontsize=11)
    ax.set_title('The model can rank its own predictions early,\nbut not at the frame the planner actually uses', fontsize=12)
    ax.set_ylim(-0.15, 1.0)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.92)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    out = OUT / 'resolution_curve.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
