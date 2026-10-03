"""Minimal drop-in replacements for the pieces of `stable_pretraining` (spt)
that `train.py` / `eval.py` use, implemented directly on top of
`transformers` + `lightning` interfaces we actually need.

We deliberately avoid depending on the real `stable_pretraining` package: it
pulls a heavy, mostly-unrelated-to-LeWM dependency chain (lightning, kornia,
timm, pandas, scikit-learn, minari, datasets, wandb, ...). For the curiosity
experiment we only ever needed two things from it:

  1. `backbone.utils.vit_hf(...)` — build a from-scratch HuggingFace
     `ViTModel` with the given size/patch/image config. Reproduced verbatim
     (same size table, same `ViTConfig` fields, same
     `interpolate_pos_encoding=True` patch) from
     `stable_pretraining/backbone/utils.py`, so the encoder architecture is
     bit-for-bit identical to the paper's.
  2. A tiny `Module`-like training wrapper — not reproduced; `train_curiosity.py`
     rolls its own plain PyTorch training loop instead of going through
     `stable_pretraining.Module` + `lightning.Trainer`, which keeps the
     dependency surface small without changing the optimization itself
     (same loss terms, same optimizer, same LR).
"""

from __future__ import annotations

from transformers import ViTConfig, ViTModel

_VIT_SIZE_CONFIGS = {
    'tiny': {'hidden_size': 192, 'num_hidden_layers': 12, 'num_attention_heads': 3},
    'small': {'hidden_size': 384, 'num_hidden_layers': 12, 'num_attention_heads': 6},
    'base': {'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12},
    'large': {'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16},
}


def vit_hf(
    size: str = 'tiny',
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
):
    """Verbatim reimplementation of `stable_pretraining.backbone.utils.vit_hf`
    (see /root/spt_src/stable_pretraining/backbone/utils.py), so a model built
    here is architecturally identical to one built with the real package."""
    if size not in _VIT_SIZE_CONFIGS:
        raise ValueError(f"Invalid size '{size}'. Choose from {list(_VIT_SIZE_CONFIGS)}")

    config_params = dict(_VIT_SIZE_CONFIGS[size])
    config_params['intermediate_size'] = config_params['hidden_size'] * 4
    config_params['image_size'] = image_size
    config_params['patch_size'] = patch_size
    config_params.update(kwargs)

    if pretrained:
        model_name = f'google/vit-{size}-patch{patch_size}-{image_size}'
        model = ViTModel.from_pretrained(
            model_name, add_pooling_layer=False, use_mask_token=use_mask_token
        )
    else:
        config = ViTConfig(**config_params)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)

    model.config.interpolate_pos_encoding = True
    return model
