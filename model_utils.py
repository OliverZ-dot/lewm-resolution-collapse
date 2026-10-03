"""Model construction helpers, extracted out of `explore_train.py` so both
it and `pretrained.py` (which needs `build_model` to load the official
checkpoints) can import them without a circular dependency."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from spt_compat import vit_hf


def load_model_cfg(cfg):
    """`config/train/model/lewm.yaml` uses `${img_size}` / `${history_size}` /
    `${embed_dim}` interpolations that Hydra normally resolves against the
    top-level train.py config via its defaults list. Since we load this file
    standalone (to guarantee byte-for-byte the same architecture regardless of
    Hydra's search-path resolution), wrap it under a container exposing those
    same top-level keys before resolving."""
    raw = OmegaConf.load(cfg.model_config_path)
    container = OmegaConf.create(
        {
            'img_size': cfg.img_size,
            'history_size': cfg.history_size,
            'embed_dim': cfg.embed_dim,
            'model': raw,
        }
    )
    OmegaConf.resolve(container)
    return container.model


def build_model(model_cfg, action_dim: int, device: str) -> JEPA:
    """Instantiate the exact architecture in config/train/model/lewm.yaml,
    substituting our dependency-free `vit_hf` / building blocks for
    `stable_pretraining`'s so the network is bit-for-bit the same shape."""
    enc = model_cfg.encoder
    encoder = vit_hf(
        size=enc.size,
        patch_size=enc.patch_size,
        image_size=enc.image_size,
        pretrained=enc.pretrained,
        use_mask_token=enc.use_mask_token,
    )

    pred = model_cfg.predictor
    predictor = ARPredictor(
        num_frames=pred.num_frames,
        input_dim=pred.input_dim,
        hidden_dim=pred.hidden_dim,
        output_dim=pred.output_dim,
        depth=pred.depth,
        heads=pred.heads,
        mlp_dim=pred.mlp_dim,
        dim_head=pred.dim_head,
        dropout=pred.dropout,
        emb_dropout=pred.emb_dropout,
    )

    act_enc_cfg = model_cfg.action_encoder
    action_encoder = Embedder(input_dim=action_dim, emb_dim=act_enc_cfg.emb_dim)

    def build_mlp(cfg):
        return MLP(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            hidden_dim=cfg.hidden_dim,
            norm_fn=torch.nn.BatchNorm1d,
        )

    projector = build_mlp(model_cfg.projector)
    pred_proj = build_mlp(model_cfg.pred_proj)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    return model.to(device)
