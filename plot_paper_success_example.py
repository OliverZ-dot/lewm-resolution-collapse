"""Figure: a real episode where the terminal-frame baseline times out
without reaching the goal and the resolution-matched fix succeeds, both
under CEM planning with the same world model, same starting state, same
goal. Frames are cropped from the 'agent' panel of the mp4s written by
`exp_success_sequential.py` (real closed-loop rollouts, not illustrations).
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

ENV = 'tworoom'
EP = 38
PAD, W, H = 16, 224, 224
C_BASE = '#8a8a8a'
C_FIX = '#2166ac'


def load_agent_frames(arm: str):
    import imageio.v2 as imageio
    path = REPO / f'outputs/runs/{ENV}/exp21_sequential/{arm}/ep_{EP}/env_0.mp4'
    frames = imageio.mimread(str(path))
    return [f[PAD:PAD + H, PAD:PAD + W] for f in frames]


def main():
    base_frames = load_agent_frames('baseline_last')
    fix_frames = load_agent_frames('fix_frame')
    n_show = 6

    base_idx = np.linspace(0, len(base_frames) - 1, n_show).round().astype(int)
    fix_idx = np.linspace(0, len(fix_frames) - 1, n_show).round().astype(int)

    fig, axes = plt.subplots(2, n_show, figsize=(2.05 * n_show, 4.55))
    for col, i in enumerate(base_idx):
        ax = axes[0, col]
        ax.imshow(base_frames[i])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C_BASE); s.set_linewidth(2.2)
        tag = 't=0 (start)' if i == 0 else (f't={i} (times out)' if col == n_show - 1 else f't={i}')
        ax.set_title(tag, fontsize=9.5)
    axes[0, 0].set_ylabel('terminal frame\n(baseline)', fontsize=11.5, fontweight='bold', color=C_BASE)

    for col, i in enumerate(fix_idx):
        ax = axes[1, col]
        ax.imshow(fix_frames[i])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C_FIX); s.set_linewidth(2.2)
        tag = 't=0 (start)' if i == 0 else (f't={i} (goal reached)' if col == n_show - 1 else f't={i}')
        ax.set_title(tag, fontsize=9.5)
    axes[1, 0].set_ylabel('resolution-matched\n(fix)', fontsize=11.5, fontweight='bold', color=C_FIX)

    fig.suptitle('Same start, same goal, same world model: a real closed-loop episode on TwoRoom',
                 fontsize=13, y=1.045)
    fig.tight_layout()
    out = OUT / 'success_example.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
