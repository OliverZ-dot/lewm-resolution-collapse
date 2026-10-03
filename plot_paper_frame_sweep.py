"""Figure: planner success rate by predicted frame (Exp 15b), with the
success-rate-blind rule's chosen frame (Exp 19) highlighted against the
terminal-frame baseline and the sweep's oracle.
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
FRAME_STAR = {'tworoom': 4, 'reacher': 4, 'pusht': 4, 'cube': 3}

BAR = '#9ecae1'
BAR_LAST = '#888888'
BAR_RULE = '#2166ac'
BAR_ORACLE_EDGE = '#d73027'


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
    for ax, env in zip(axes, ENVS):
        sw = json.loads((REPO / f'outputs/runs/{env}/exp15_frame_sweep/sweep.json').read_text())
        arms = sw['arms']
        frame_idxs = sw['frame_idxs']
        srs = [arms[f'frame_{i}']['sr'] for i in frame_idxs]
        colors = [BAR] * len(frame_idxs)
        colors[-1] = BAR_LAST  # frame_H == last
        rule_i = FRAME_STAR[env]
        rule_pos = frame_idxs.index(rule_i)
        colors[rule_pos] = BAR_RULE
        oracle_pos = int(np.argmax(srs))

        xs = np.arange(len(frame_idxs))
        bars = ax.bar(xs, srs, color=colors, edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
        bars[oracle_pos].set_edgecolor(BAR_ORACLE_EDGE)
        bars[oracle_pos].set_linewidth(2.6)

        for x, v in zip(xs, srs):
            ax.text(x, v + 1.6, f'{v:.0f}', ha='center', fontsize=8.5, color='#333333')

        ax.set_xticks(xs)
        ax.set_xticklabels([f'$t_{i}$' if i != frame_idxs[-1] else f'$t_{i}$=last' for i in frame_idxs],
                            fontsize=9.5)
        ax.set_title(LABEL[env], fontsize=12.5)
        ax.set_ylim(0, 108)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, lw=0.6, zorder=0)
    axes[0].set_ylabel('planner success rate (%)', fontsize=11)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BAR_LAST, label='terminal frame (last, the baseline)'),
        plt.Rectangle((0, 0), 1, 1, fc=BAR_RULE, label='rule-selected frame (Exp 19, blind to success rate)'),
        plt.Rectangle((0, 0), 1, 1, fc=BAR, ec=BAR_ORACLE_EDGE, lw=2.2, label='oracle best of the sweep'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Success rate by which predicted frame the planner asks about', fontsize=13.5, y=1.03)
    fig.tight_layout()
    out = OUT / 'frame_sweep_rule.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
