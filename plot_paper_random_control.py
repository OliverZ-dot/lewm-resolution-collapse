"""Figure: random-cost immunity checks (Exp 17, Exp 20). Two panels:
  (a) additive control -- MultiHorizon (mid + last + L2) vs. a spread-matched
      random term added to the same base cost
  (b) replacement control -- the rule-selected frame vs. a spread-matched
      random cost used as the entire main cost
Both compared against the terminal-frame (`last`) baseline.
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

C_LAST = '#888888'
C_REAL = '#2166ac'
C_RAND = '#d73027'


def panel(ax, values, title, ylab=False):
    n = len(ENVS)
    w = 0.26
    xs = np.arange(n)
    ax.bar(xs - w, [values[e]['last'] for e in ENVS], width=w, color=C_LAST, label='terminal frame (last)')
    ax.bar(xs, [values[e]['real'] for e in ENVS], width=w, color=C_REAL, label='real term')
    ax.bar(xs + w, [values[e]['rand'] for e in ENVS], width=w, color=C_RAND, label='spread-matched noise')
    for i, e in enumerate(ENVS):
        for off, key in zip((-w, 0, w), ('last', 'real', 'rand')):
            v = values[e][key]
            ax.text(i + off, v + 1.5, f'{v:.0f}', ha='center', fontsize=7.6, color='#333333')
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[e] for e in ENVS], fontsize=10.5)
    ax.set_title(title, fontsize=11.8)
    ax.set_ylim(0, 108)
    if ylab:
        ax.set_ylabel('planner success rate (%)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, lw=0.6, zorder=0)


def main():
    add_vals = {}
    for env in ENVS:
        d = json.loads((REPO / f'outputs/runs/{env}/exp17_multi_h_control/control.json').read_text())
        add_vals[env] = {
            'last': d['arms']['last']['sr'],
            'real': d['arms']['multi_h_l2']['sr'],
            'rand': d['arms']['randmid_l2']['sr'],
        }

    rep_vals = {}
    for env in ENVS:
        d = json.loads((REPO / f'outputs/runs/{env}/exp20_frame_rule_control/control.json').read_text())
        rep_vals[env] = {
            'last': d['arms']['last']['sr'],
            'real': d['arms']['frame_star']['sr'],
            'rand': d['arms']['rand_frame_star']['sr'],
        }

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    panel(axes[0], add_vals, '(a) Additive: mid-horizon term stacked\non the terminal-frame cost (Exp 17)', ylab=True)
    panel(axes[1], rep_vals, '(b) Replacement: rule-selected frame\nreplaces the terminal-frame cost (Exp 20)')
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_LAST, label='terminal frame (last)'),
        plt.Rectangle((0, 0), 1, 1, fc=C_REAL, label='real term'),
        plt.Rectangle((0, 0), 1, 1, fc=C_RAND, label='spread-matched random noise'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle('The gain is not a magnitude artifact: random costs of the same spread do not reproduce it',
                 fontsize=13, y=1.04)
    fig.tight_layout()
    out = OUT / 'random_control.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
