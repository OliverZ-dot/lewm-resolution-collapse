"""Real-dataset evaluation protocol, extracted from the paper's own
`eval.py` so `explore_train.py` (SIGReg-as-Curiosity conditions) and
`eval_lewm_baseline.py` (official pretrained checkpoints) can both evaluate
against it.

Why this exists: the harness originally evaluated against a small
synthetic probe pool built on the fly with `WeakPolicy` (see
`explore_train.build_eval_probe_set`), because the real expert HDF5
datasets weren't available locally. That understated performance in two
ways relative to `eval.py`'s actual protocol:

1. Pool size/coverage: 30 synthetic episodes vs. the real dataset's
   10,000 (tworoom) / 18,685 (pusht) episodes -- far less diverse, and not
   necessarily in the same state-visitation distribution the pretrained
   encoder/predictor were fit on.
2. Missing normalization: `eval.py` fits a per-column `StandardScaler` on
   `action`/`proprio`/`state` from the real dataset and passes it to
   `WorldModelPolicy(process=...)`, so CEM's `var_scale=1.0` Gaussian
   sampling operates in *z-scored* action units, not raw `[-1, 1]` Box
   units. Measured directly on `pusht_expert_train.h5`: real action std is
   ~0.21, not 1.0 -- so without this, CEM was searching ~5x too coarse
   relative to what the model was ever trained to predict accurately over.

Both real datasets are downloaded (via the HF mirror) and decompressed
into `$STABLEWM_HOME/datasets/{tworoom,pusht_expert_train}.h5` -- see
`config/eval/{tworoom,pusht}.yaml`'s `dataset_name` for the exact names
this harness's own eval.py expects.
"""

from __future__ import annotations

import numpy as np
from sklearn import preprocessing

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


def _episode_col(dataset) -> str:
    return 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'


def get_episodes_length(dataset, episodes) -> np.ndarray:
    """Behavior-preserving copy of `eval.py`'s helper of the same name."""
    col_name = _episode_col(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def load_real_dataset(dataset_name: str, keys_to_cache: list[str]) -> HDF5Dataset:
    """Load `$STABLEWM_HOME/datasets/{dataset_name}.h5`. `pixels` is
    intentionally left out of `keys_to_cache` (as in `eval.py`) -- it's
    read lazily per-episode via `load_chunk`, since eagerly caching all
    frames of a 10k+-episode dataset would be tens of GB of RAM."""
    return HDF5Dataset(
        dataset_name,
        keys_to_cache=keys_to_cache,
        cache_dir=swm.data.utils.get_cache_dir(),
    )


def build_process(dataset, keys_to_cache: list[str]) -> dict:
    """Fit one `StandardScaler` per non-pixel cached column on the real
    dataset's actual values -- behavior-preserving copy of `eval.py`'s
    `process` construction (lines ~71-83), including the `goal_{col}`
    alias so `WorldModelPolicy._prepare_info` normalizes the goal columns
    the same way as the live observation columns."""
    process = {}
    for col in keys_to_cache:
        if col == 'pixels':
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]
    return process


def build_real_eval_pairs(
    dataset, cfg, num_eval: int, goal_offset_steps: int, seed: int
) -> tuple[list[int], list[int]]:
    """Behavior-preserving copy of `eval.py`'s (episode, start_step)
    sampling (lines ~108-134): pick `num_eval` random valid starting
    points -- valid meaning `start_step + goal_offset_steps` still falls
    inside that episode -- from across the *entire* real dataset."""
    col_name = _episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(seed)
    random_row_pos = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    random_row_pos = np.sort(valid_indices[random_row_pos])

    rows = dataset.get_row_data(random_row_pos)
    episodes_idx = rows[col_name].tolist()
    start_steps = rows['step_idx'].tolist()
    return episodes_idx, start_steps


def build_real_eval(cfg, dataset_name: str, keys_to_cache: list[str], seed: int):
    """One-stop entry point: load the real dataset, fit the process
    normalizers, and draw the fixed (episodes_idx, start_steps) eval pairs.

    Returns (dataset, process, episodes_idx, start_steps).
    """
    dataset = load_real_dataset(dataset_name, keys_to_cache)
    process = build_process(dataset, keys_to_cache)
    episodes_idx, start_steps = build_real_eval_pairs(
        dataset,
        cfg,
        num_eval=cfg.loop.eval_episodes,
        goal_offset_steps=cfg.eval.goal_offset_steps,
        seed=seed,
    )
    return dataset, process, episodes_idx, start_steps
