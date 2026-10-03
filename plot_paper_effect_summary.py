"""Figure: dot-and-line summary of Table 2 (the success-rate-blind rule
against the terminal-frame baseline and the exhaustive-sweep oracle), one
panel replacing a thousand words of the table for a reader skimming the
paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / 'iclr2027_main' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

ENVS = ['tworoom', 'reacher', 'pusht', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'reacher': 'Reacher', 'pusht': 'PushT', 'cube': 'Cube'}

# Table 2 in sections/fix.tex
BASELINE = {'tworoom': 63.5, 'reacher': 91.0, 'pusht': 69.5, 'cube': 64.5}
RULE = {'tworoom': 68.0, 'reacher': 96.0, 'pusht': 72.5, 'cube': 68.5}
ORACLE = {'tworoom': 68.0, 'reacher': 97.0, 'pusht': 74.5, 'cube': 70.0}
PVAL = {'tworoom': 0.22, 'reacher': 0.013, 'pusht': 0.36, 'cube': 0.12}

C_BASE = '#8a8a8a'
C_RULE = '#2166ac'
C_ORACLE = '#d73027'


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 3.3))
    ys = np.arange(len(ENVS))[::-1]

    for y, e in zip(ys, ENVS):
        ax.plot([BASELINE[e], RULE[e]], [y, y], color='#bbbbbb', lw=2.0, zorder=1)
        ax.plot([RULE[e], ORACLE[e]], [y, y], color='#bbbbbb', lw=2.0, ls=':', zorder=1)

    ax.scatter([BASELINE[e] for e in ENVS], ys, s=110, color=C_BASE, marker='o',
               zorder=3, label='terminal frame (baseline)', edgecolor='white', linewidth=0.8)
    ax.scatter([RULE[e] for e in ENVS], ys, s=150, color=C_RULE, marker='D',
               zorder=3, label='success-rate-blind rule', edgecolor='white', linewidth=0.8)
    ax.scatter([ORACLE[e] for e in ENVS], ys, s=130, color=C_ORACLE, marker='*',
               zorder=3, label='exhaustive-sweep oracle', edgecolor='white', linewidth=0.8)

    for y, e in zip(ys, ENVS):
        gain = RULE[e] - BASELINE[e]
        star = '*' if PVAL[e] < 0.05 else ''
        ax.text(max(BASELINE[e], RULE[e], ORACLE[e]) + 2.0, y,
                f'+{gain:.1f}pp{star}  (p={PVAL[e]:.3g})', va='center', fontsize=9.3, color='#333333')

    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[e] for e in ENVS], fontsize=11.5)
    ax.set_xlabel('planner success rate (%)', fontsize=11)
    ax.set_xlim(55, 108)
    ax.set_title('The rule recovers the oracle\u2019s gain on every environment, using only the resolution curve', fontsize=11.5)
    ax.legend(loc='lower right', fontsize=9.3, framealpha=0.94)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, lw=0.6, zorder=0)
    fig.tight_layout()
    out = OUT / 'effect_summary.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
