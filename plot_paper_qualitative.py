"""Qualitative gallery for the ICLR paper: real rendered frames from all
four environments, showing one representative init -> goal rollout each at
the same six checkpoints (t0, t5, t10, t15, t20, t25) the resolution-curve
experiments (Exp 6/13/15a) measure. Purely illustrative: these are ground
truth simulator frames from the expert dataset, not model reconstructions
(the model has no pixel decoder).

Saves:
  figures/qual_gallery.png   -- 4 rows (env) x 6 cols (checkpoint)
  figures/qual_<env>.png     -- one env at a time, larger, for the appendix
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from real_eval import build_real_eval

REPO = Path(__file__).resolve().parent
OUT = REPO / 'iclr2027' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

ENVS = ['tworoom', 'reacher', 'pusht', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'reacher': 'Reacher', 'pusht': 'PushT', 'cube': 'Cube'}
PICK_IDX = {'tworoom': 2, 'reacher': 0, 'pusht': 1, 'cube': 0}


def to_img(t):
    arr = t.numpy() if hasattr(t, 'numpy') else np.asarray(t)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) if arr.max() <= 1.5 else np.clip(arr / 255.0, 0, 1)
    return arr


def load_storyboard(env: str, pick: int):
    cfg = OmegaConf.load(f'config/explore/{env}.yaml')
    OmegaConf.set_struct(cfg, False)
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    goal_offset = int(cfg.eval.goal_offset_steps)
    ep, start = episodes_idx[pick], start_steps[pick]
    sl = dataset._load_slice(ep, start, start + goal_offset + 1)
    pixels = sl['pixels']
    marks = list(range(0, goal_offset + 1, 5))
    frames = [to_img(pixels[m]) for m in marks]
    return frames, marks


def main():
    all_frames = {}
    all_marks = {}
    for env in ENVS:
        frames, marks = load_storyboard(env, PICK_IDX[env])
        all_frames[env] = frames
        all_marks[env] = marks
        print(f'{env}: {len(frames)} frames at steps {marks}')

    ncols = max(len(v) for v in all_frames.values())
    fig, axes = plt.subplots(len(ENVS), ncols, figsize=(2.05 * ncols, 2.25 * len(ENVS)))
    for row, env in enumerate(ENVS):
        frames, marks = all_frames[env], all_marks[env]
        for col in range(ncols):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(frames):
                ax.imshow(frames[col])
                tag = 'init' if marks[col] == 0 else (f'goal ({marks[col]})' if col == len(frames) - 1 else f't={marks[col]}')
                ax.set_title(tag, fontsize=10)
        axes[row, 0].axis('on')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for s in axes[row, 0].spines.values():
            s.set_visible(False)
        axes[row, 0].set_ylabel(LABEL[env], fontsize=13, fontweight='bold', labelpad=10)
    fig.suptitle('Example rollouts across the four benchmark environments', y=1.01, fontsize=14)
    fig.tight_layout()
    out = OUT / 'qual_gallery.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')

    # one larger per-env figure for the appendix
    for env in ENVS:
        frames, marks = all_frames[env], all_marks[env]
        fig, axes = plt.subplots(1, len(frames), figsize=(2.6 * len(frames), 2.9))
        for col, (im, m) in enumerate(zip(frames, marks)):
            axes[col].imshow(im)
            axes[col].axis('off')
            tag = 'init (t=0)' if m == 0 else (f'goal (t={m})' if col == len(frames) - 1 else f't={m}')
            axes[col].set_title(tag, fontsize=11)
        fig.suptitle(f'{LABEL[env]}: example rollout, frames at each planning checkpoint', y=1.03, fontsize=12)
        fig.tight_layout()
        out = OUT / f'qual_{env}.png'
        fig.savefig(out, dpi=170, bbox_inches='tight')
        plt.close(fig)
        print(f'saved {out}')


if __name__ == '__main__':
    main()
