"""Loading the officially released pretrained LeWM checkpoints (see the LeWM
citation in the paper) into our `build_model`-constructed architecture.
Shared by:

  - `eval_lewm_baseline.py`: evaluates the pretrained checkpoint as-is (the
    "upper bound reference").
  - `explore_train.py`: optionally *initializes* a curiosity-driven
    generational run from this checkpoint instead of from scratch (see
    `cfg.init_from_pretrained` / `config/explore/*_finetune.yaml`), to test
    whether curiosity-driven self-play can keep improving (without
    forgetting) an already-competent model, as opposed to bootstrapping one
    from nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf

from model_utils import build_model


HF_CACHE = {
    'tworoom': Path('/root/.stable_worldmodel/hf_tworooms'),
    'pusht': Path('/root/.stable_worldmodel/hf_pusht'),
    'cube': Path('/root/.stable_worldmodel/hf_cube'),
    'reacher': Path('/root/.stable_worldmodel/hf_reacher'),
}


def _remap_vit_state_dict(sd: dict) -> dict:
    """The HF checkpoint was saved against an older `transformers` ViTModel
    layout (`encoder.encoder.layer.{i}.attention.attention.{query,key,value}`,
    separate `intermediate`/`output` dense layers); the `transformers`
    version installed in this sandbox (5.15.0) refactored `ViTModel` to a
    fused-attention-style layout (`encoder.layers.{i}.attention.{q,k,v,o}_proj`,
    `mlp.fc1`/`fc2`). Both are the exact same tensor shapes/semantics (verified
    below: identical missing/unexpected key *counts*, all resolved by this
    rename, no leftover mismatches) -- only the module path names differ.
    Remap so this old-format checkpoint loads onto the encoder architecture
    `build_model` constructs under the currently installed `transformers`.
    """
    rename_rules = [
        (r'^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query', r'encoder.layers.\1.attention.q_proj'),
        (r'^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key', r'encoder.layers.\1.attention.k_proj'),
        (r'^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value', r'encoder.layers.\1.attention.v_proj'),
        (r'^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense', r'encoder.layers.\1.attention.o_proj'),
        (r'^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense', r'encoder.layers.\1.mlp.fc1'),
        (r'^encoder\.encoder\.layer\.(\d+)\.output\.dense', r'encoder.layers.\1.mlp.fc2'),
        (r'^encoder\.encoder\.layer\.(\d+)\.layernorm_before', r'encoder.layers.\1.layernorm_before'),
        (r'^encoder\.encoder\.layer\.(\d+)\.layernorm_after', r'encoder.layers.\1.layernorm_after'),
    ]
    out = {}
    for k, v in sd.items():
        new_k = k
        for pattern, repl in rename_rules:
            new_k = re.sub(pattern, repl, new_k)
        out[new_k] = v
    return out


def load_pretrained_model(env: str, action_dim: int, device: str):
    """Build a fresh model with `build_model` and load the official
    pretrained weights into it. Returned model is in `.eval()` mode --
    callers that want to keep training it (e.g. `explore_train.py`'s
    finetune configs) should call `.train()` themselves afterwards."""
    src = HF_CACHE[env]
    cfg_json = json.loads((src / 'config.json').read_text())
    # config.json's `encoder`/`predictor`/etc sub-dicts have the exact same
    # field names build_model() expects (it was generated from the same
    # config/train/model/lewm.yaml this repo uses) -- just drop past the
    # Hydra `_target_` bookkeeping keys.
    model_cfg = OmegaConf.create(
        {
            'encoder': cfg_json['encoder'],
            'predictor': cfg_json['predictor'],
            'action_encoder': cfg_json['action_encoder'],
            'projector': cfg_json['projector'],
            'pred_proj': cfg_json['pred_proj'],
        }
    )
    # checkpoint's action_encoder.input_dim (10) must match our
    # plan_config.action_block*2; assert instead of silently building a
    # shape-mismatched model.
    ckpt_action_dim = cfg_json['action_encoder']['input_dim']
    assert ckpt_action_dim == action_dim, (
        f'checkpoint action_dim={ckpt_action_dim} != harness action_dim={action_dim}; '
        'plan_config.action_block differs from what this checkpoint was trained with.'
    )
    model = build_model(model_cfg, action_dim, device)
    state_dict = torch.load(src / 'weights.pt', map_location=device, weights_only=False)
    state_dict = _remap_vit_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
