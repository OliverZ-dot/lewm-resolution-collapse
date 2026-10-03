"""Curiosity-JEPA generational collect -> train -> eval loop.

Fair-comparison harness for LeWM_CuriosityJEPA_design.md ("SIGReg-as-Curiosity"):
holds the model architecture (config/train/model/lewm.yaml), optimizer,
SIGReg weight, planner (CEM) settings, and total env-interaction / gradient-
step budget fixed, and varies only *which policy collects the training
data* across conditions:

  exploration.name = random        -- baseline (no world-model-driven exploration)
  exploration.name = disagreement  -- ensemble-disagreement intrinsic reward
  exploration.name = novelty       -- SIGReg-projection novelty intrinsic reward
  exploration.name = combined      -- disagreement + novelty

Usage:
    python explore_train.py exploration.name=disagreement
    python explore_train.py exploration.name=random
    python explore_train.py exploration.name=novelty
    python explore_train.py exploration.name=combined

Each run is fully independent (fresh model, fresh buffer, same seed unless
overridden) so the four conditions can be launched in parallel and compared
on identical episode/gradient-step budgets.
"""

from __future__ import annotations

import os

# Must happen before numpy/torch import: 4 independent `exploration.name`
# conditions are launched as separate processes sharing one GPU+host, and
# each of torch/OpenMP/MKL defaults to sizing its thread pool off the
# *whole machine's* core count, unaware of its siblings. Left unset, 4
# processes x 64 threads each = 256+ threads fight over the host's cores,
# and simple CPU-bound work (TwoRoom env stepping/rendering) slows down by
# an order of magnitude from context-switch thrashing alone. This is set
# on `os.environ` (not `torch.set_num_threads`, which only limits the intra-op
# pool and can't be lowered after first use) and is intentionally
# conservative -- this codebase's operations are mostly small
# (embed_dim=192), so more threads past this stop paying for themselves.
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
# Reacher/Cube (config/explore/{reacher,cube}.yaml) are MuJoCo-rendered envs
# and need an offscreen GL backend on a headless host; matches eval.py's own
# setting. No-op for TwoRoom/PushT (pygame-rendered, no MuJoCo involved).
os.environ.setdefault('MUJOCO_GL', 'egl')

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

torch.set_num_threads(8)

import stable_worldmodel as swm

from jepa import JEPA
from module import SIGReg
from curiosity import EnsembleDynamics, RunningProjectionDensity, IntrinsicObjective
from real_eval import build_real_eval
from model_utils import build_model, load_model_cfg
from pretrained import load_pretrained_model


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """(..., H, W, C) uint8 -> (..., C, H, W) float32, ImageNet-normalized.

    Mirrors `utils.get_img_preprocessor` (ToImage w/ ImageNet stats + Resize)
    from the original train.py, minus the Resize since `swm.World`'s
    MegaWrapper already renders pixels at `image_shape == (img_size, img_size)`.
    The `ReplayBuffer`/DataLoader path returns raw HWC uint8 pixels (unlike
    `WorldModelPolicy`'s own `transform` dict, which the planner applies to
    pixels internally), so this must be applied explicitly before `model.encode`.
    """
    pixels = pixels.float() / 255.0
    pixels = pixels.movedim(-1, -3)  # (..., C, H, W)
    extra_dims = pixels.ndim - 3
    mean = _IMAGENET_MEAN.to(pixels.device).view(*([1] * extra_dims), 3, 1, 1)
    std = _IMAGENET_STD.to(pixels.device).view(*([1] * extra_dims), 3, 1, 1)
    return (pixels - mean) / std


def lewm_train_step(model: JEPA, sigreg: SIGReg, batch: dict, cfg, ensemble=None):
    """Standard LeWM loss (verbatim from train.py's `lejepa_forward`),
    optionally + the ensemble heads' bootstrap loss. `batch['pixels']` is
    (B, T, C, H, W) with T = history_size + num_preds."""
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    batch['pixels'] = _normalize_pixels(batch['pixels'])
    output = model.encode(batch)

    emb = output['emb']
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + cfg.loss.sigreg.weight * sigreg_loss

    logs = {'pred_loss': pred_loss.item(), 'sigreg_loss': sigreg_loss.item()}

    if ensemble is not None:
        ens_loss = ensemble.ensemble_loss(emb.detach(), act_emb.detach())
        loss = loss + ens_loss
        logs['ensemble_loss'] = ens_loss.item()

    logs['loss'] = loss.item()
    return loss, logs, emb.detach()


def make_exploration_policy(cfg, model: JEPA, device: str):
    """Build the world-model policy used to *collect* data this run.

    `random` skips the world model entirely (RandomPolicy). All other
    conditions plan with CEM against an `IntrinsicObjective` composed with
    an `EnsembleDynamics` wrapper around the *same* model instance that is
    being trained (so exploration always reflects the current epistemic
    state, Plan2Explore-style).
    """
    if cfg.exploration.name == 'random':
        return swm.policy.RandomPolicy(seed=cfg.seed), None

    ens_cfg = cfg.exploration
    pred_cfg = load_model_cfg(cfg).predictor
    predictor_kwargs = dict(
        num_frames=pred_cfg.num_frames,
        input_dim=pred_cfg.input_dim,
        hidden_dim=pred_cfg.hidden_dim,
        output_dim=pred_cfg.output_dim,
        depth=pred_cfg.depth,
        heads=pred_cfg.heads,
        mlp_dim=pred_cfg.mlp_dim,
        dim_head=pred_cfg.dim_head,
        dropout=pred_cfg.dropout,
        emb_dropout=pred_cfg.emb_dropout,
    )
    ensemble = EnsembleDynamics(
        model,
        num_heads=ens_cfg.num_heads,
        predictor_kwargs=predictor_kwargs,
        bootstrap_prob=ens_cfg.bootstrap_prob,
        history_size=cfg.history_size,
    ).to(device)

    w_dis = ens_cfg.w_disagreement if ens_cfg.name in ('disagreement', 'combined') else 0.0
    w_nov = ens_cfg.w_novelty if ens_cfg.name in ('novelty', 'combined') else 0.0
    # Bugfix: the base yaml's `exploration.w_novelty: 0.0` is only meant as
    # the *inert default* for conditions that don't use novelty at all
    # (random/disagreement); a bare `novelty` run was silently inheriting
    # that same 0.0 (only `combined` got the `or 1.0` fallback below),
    # making the "novelty" condition run with a flat/zero intrinsic cost.
    # Apply the fallback per-condition instead of only for 'combined'.
    if ens_cfg.name in ('disagreement', 'combined'):
        w_dis = w_dis or 1.0
    if ens_cfg.name in ('novelty', 'combined'):
        w_nov = w_nov or 1.0

    density = RunningProjectionDensity(
        dim=cfg.embed_dim,
        num_proj=ens_cfg.novelty.num_proj,
        num_bins=ens_cfg.novelty.num_bins,
        value_range=ens_cfg.novelty.value_range,
        seed=cfg.seed,
    ).to(device)

    objective = IntrinsicObjective(
        density, w_disagreement=w_dis, w_novelty=w_nov, context_len=cfg.history_size
    )
    # Curiosity-driven exploration has no extrinsic goal: skip
    # `default_goal_encode` (which otherwise asserts `'goal' in info_dict`).
    evaluator = swm.planning.ShootingCostEvaluator(
        model=ensemble, objective=objective, encode_goal=None
    )

    solver = swm.planning.CEMSolver(
        cost=evaluator,
        num_samples=cfg.solver.num_samples,
        var_scale=cfg.solver.var_scale,
        n_steps=cfg.solver.n_steps,
        topk=cfg.solver.topk,
        device=device,
        seed=cfg.seed,
    )
    plan_config = swm.PlanConfig(**cfg.plan_config)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        transform={'pixels': _pixel_transform(cfg.img_size)},
    )
    return policy, (ensemble, density)


class _TorchPixelReplayBuffer(swm.data.ReplayBuffer):
    """`World._evaluate_from_dataset`'s `_extract_init_goal` was written
    against Lance-backed datasets and calls `.permute(...)` on pixel columns
    from `load_chunk`, which assumes torch tensors; the in-memory
    `ReplayBuffer` returns plain numpy there. Only `_load_slice` (used by
    `load_chunk`/`load_episode`, i.e. the dataset-driven eval path) needs the
    fixup -- `__getitem__` (the DataLoader/training path) goes through
    `_gather_clip` instead and is untouched.
    """

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        slc = super()._load_slice(ep_idx, start, end)
        for col in slc:
            if col.startswith('pixels'):
                # `_extract_init_goal` un-permutes pixel columns with
                # `.permute(0, 2, 3, 1)` assuming the on-disk (T, C, H, W)
                # layout the original Lance-backed dataset stores; our raw
                # env pixels are (T, H, W, C), so pre-permute to match.
                slc[col] = torch.from_numpy(slc[col]).permute(0, 3, 1, 2).contiguous()
        return slc


def build_eval_probe_set(cfg):
    """Collect a *fixed* held-out set of (episode, start_step) pairs via
    `RandomPolicy`, seeded identically regardless of `exploration.name` so
    every condition (and every generation within a condition) is evaluated
    on the exact same init-state/goal-state pairs. Mirrors the original
    dataset-driven protocol in config/eval/tworoom.yaml (`world.evaluate(
    dataset=..., episodes_idx=..., start_steps=..., goal_offset=...)`), just
    with the "dataset" generated on the fly instead of loaded from disk,
    since we don't ship a pre-collected TwoRoom HDF5 dataset in this sandbox.
    """
    probe_buffer = _TorchPixelReplayBuffer(
        max_steps=cfg.eval.num_probe_episodes * cfg.eval.probe_max_steps,
        history_len=cfg.history_size + cfg.num_preds,
        # frameskip=1 (not cfg.frameskip): this buffer is only ever read via
        # `load_chunk`/`_extract_init_goal` (raw per-step indexing, matching
        # the original eval.py's dataset convention where `action_block`
        # governs replanning cadence but the on-disk dataset is unstrided),
        # never through the strided `sample()` path where frameskip matters.
        frameskip=1,
    )
    env_kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True) if 'env_kwargs' in cfg else {}
    probe_world = swm.World(
        cfg.env_name,
        num_envs=1,
        image_shape=(cfg.img_size, cfg.img_size),
        max_episode_steps=cfg.eval.probe_max_steps,
        **env_kwargs,
    )
    probe_world.set_policy(swm.policy.RandomPolicy(seed=cfg.loop.eval_seed))
    probe_world.collect(
        writer=probe_buffer,
        episodes=cfg.eval.num_probe_episodes,
        seed=cfg.loop.eval_seed,
        progress=False,
    )
    probe_world.close()

    rng = np.random.default_rng(cfg.loop.eval_seed)
    lengths = probe_buffer.lengths
    valid_eps = np.where(lengths > cfg.eval.goal_offset_steps)[0]
    if len(valid_eps) == 0:
        raise RuntimeError(
            'build_eval_probe_set: no probe episode is longer than '
            f'eval.goal_offset_steps={cfg.eval.goal_offset_steps}; raise '
            'eval.probe_max_steps or lower goal_offset_steps.'
        )
    n_eval = min(cfg.loop.eval_episodes, len(valid_eps) * 4)
    pairs = []
    for _ in range(n_eval):
        ep = int(rng.choice(valid_eps))
        max_start = int(lengths[ep]) - cfg.eval.goal_offset_steps - 1
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        pairs.append((ep, start))
    return probe_buffer, pairs


def evaluate_on_probe_set(world, probe_buffer, eval_pairs, eval_policy, cfg):
    """Run `eval_policy` on every fixed (episode, start) pair and return the
    aggregate success rate. Uses `swm.World.evaluate`'s dataset-driven mode
    one episode at a time (our `world` has `num_envs == 1`)."""
    world.set_policy(eval_policy)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    successes = []
    with torch.no_grad():
        for ep, start in eval_pairs:
            result = world.evaluate(
                dataset=probe_buffer,
                episodes_idx=[ep],
                start_steps=[start],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                callables=callables,
            )
            successes.append(bool(result['episode_successes'][0]))
    return {'success_rate': 100.0 * (sum(successes) / len(successes) if successes else 0.0)}


def make_eval_policy(cfg, model: JEPA, device: str, process: dict | None = None, objective=None, callbacks=None):
    """Standard LeWM goal-reaching planner (GoalMSE), used identically
    across all `exploration` conditions to measure success rate.

    `process`: optional dict of per-column `StandardScaler`s (see
    `real_eval.build_process`), matching `eval.py`'s own protocol -- CEM's
    `var_scale=1.0` Gaussian sampling then perturbs actions in z-scored
    units (real dataset action std ~0.21, not the raw Box's implicit 1.0)
    instead of raw `[-1, 1]` units, and `WorldModelPolicy.get_action`
    inverse-transforms the solved action back before it's executed.

    `objective`: optional override of the planning cost (default:
    `swm.planning.GoalMSE()`). Used to swap in
    `swm.planning.WeightedSum([(1.0, GoalMSE()), (lam, extra_term)])`
    without touching the rest of the eval harness.

    `callbacks`: optional list of `stable_worldmodel.planning.solver.callbacks.Callback`
    forwarded verbatim to `CEMSolver` -- used by the Idea 1 mechanism probe
    (`mechanism_probe.py`) to record the raw per-candidate cost distribution
    at each CEM iteration (does not affect which action is planned/executed,
    purely observational)."""
    if objective is None:
        objective = swm.planning.GoalMSE()
    evaluator = swm.planning.ShootingCostEvaluator(model=model, objective=objective)
    solver = swm.planning.CEMSolver(
        cost=evaluator,
        num_samples=cfg.solver.num_samples,
        var_scale=cfg.solver.var_scale,
        n_steps=cfg.solver.n_steps,
        topk=cfg.solver.topk,
        device=device,
        seed=cfg.seed,
        callbacks=callbacks,
    )
    plan_config = swm.PlanConfig(**cfg.plan_config)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        transform={'pixels': _pixel_transform(cfg.img_size), 'goal': _pixel_transform(cfg.img_size)},
        process=process,
    )


def _pixel_transform(img_size: int):
    """Per-image transform handed to `WorldModelPolicy`/`BasePolicy`, which
    internally permutes HWC->CHW and wraps each frame as a `tv_tensors.Image`
    before calling this. Matches `utils.get_img_preprocessor` (ToImage w/
    ImageNet stats, then Resize) used by the original train.py/eval.py."""
    from torchvision.transforms import v2 as T

    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN.tolist(), std=_IMAGENET_STD.tolist()),
            T.Resize(size=img_size),
        ]
    )


def save_buffer_snapshot(buffer: swm.data.ReplayBuffer, path: Path) -> None:
    """Persist full episode contents (pixels included) via plain pickling
    through `torch.save`, rather than `buffer.dump(..., format=...)`'s
    Format writers (`folder` writes one JPEG per frame -- millions of tiny
    files across 4 conditions x 8 generations; `lance`/`lance_video` add a
    columnar-store round-trip we don't need). This is for *post-hoc
    analysis without re-running collection* -- e.g. `torch.load(path)`
    yields the same `list[dict[str, list[np.ndarray]]]` shape
    `ReplayBuffer.episodes()` produces, loadable regardless of whether
    `stable-worldmodel` is even installed in the future analysis env.
    """
    tmp = path.with_suffix('.tmp')
    torch.save(list(buffer.episodes()), tmp)
    tmp.replace(path)  # atomic-ish swap so a crash mid-write can't corrupt the last good snapshot


def save_checkpoint(
    path: Path, gen: int, model: JEPA, optimizer, extras, global_step: int
) -> None:
    state = {
        'generation': gen,
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if extras is not None:
        state['ensemble_predictors'] = extras[0].ensemble_predictors.state_dict()
        state['density'] = extras[1].state_dict()
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


@hydra.main(version_base=None, config_path='./config/explore', config_name='tworoom')
def run(cfg):
    device = cfg.device
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)
    metrics_path = out_dir / 'metrics.jsonl'
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Generalized from the original hardcoded `* 2` (TwoRoom/PushT/Reacher's
    # `action_space.shape == (2,)`) so Cube (`action_space.shape == (5,)`,
    # see config/explore/cube.yaml's `env_action_dim: 5`) builds an
    # action_encoder with the right input width instead of silently
    # truncating/misaligning the action vector.
    action_dim = cfg.plan_config.action_block * cfg.get('env_action_dim', 2)
    model = build_model(load_model_cfg(cfg), action_dim, device)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

    explore_policy, extras = make_exploration_policy(cfg, model, device)
    eval_policy = None  # built lazily once model has been trained a bit

    buffer = swm.data.ReplayBuffer(
        max_steps=cfg.loop.buffer_max_steps,
        history_len=cfg.history_size + cfg.num_preds,
        frameskip=cfg.frameskip,
    )

    # Guarded with `'env_kwargs' in cfg` rather than `cfg.get('env_kwargs', {})`:
    # OmegaConf's `.get()` fallback returns a plain Python dict (not a
    # DictConfig), which `OmegaConf.to_container` rejects with "Input cfg is
    # not an OmegaConf config object". Needed by Reacher
    # (`env_kwargs: {task: qpos_match}`, see config/explore/reacher.yaml);
    # a no-op {} for TwoRoom/PushT/Cube, which don't set `env_kwargs` at all.
    env_kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True) if 'env_kwargs' in cfg else {}
    world = swm.World(
        cfg.env_name,
        num_envs=1,
        image_shape=(cfg.img_size, cfg.img_size),
        max_episode_steps=cfg.loop.max_episode_steps,
        **env_kwargs,
    )

    optimizer_params = list(model.parameters())
    if extras is not None:
        optimizer_params += list(extras[0].ensemble_predictors.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )

    # -- resume from the last checkpoint this run dir has, if any ---------
    # Unattended multi-hour runs will eventually hit a transient crash (OOM,
    # a flaky env step, host hiccup); a supervisor restarting this same
    # command should continue from the last completed generation instead of
    # silently re-running from generation 0 every time.
    global_step = 0
    start_gen = 0
    existing_ckpts = sorted(ckpt_dir.glob('gen_*.pt'))
    if existing_ckpts:
        ckpt = torch.load(existing_ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if extras is not None and 'ensemble_predictors' in ckpt:
            extras[0].ensemble_predictors.load_state_dict(ckpt['ensemble_predictors'])
            extras[1].load_state_dict(ckpt['density'])
        global_step = ckpt['global_step']
        start_gen = ckpt['generation'] + 1
        print(
            f'[resume] loaded {existing_ckpts[-1].name}: resuming at '
            f'generation {start_gen} (global_step={global_step})'
        )

    buffer_snapshot_path = out_dir / 'buffer_latest.pt'
    if start_gen > 0 and buffer_snapshot_path.exists():
        for ep in torch.load(buffer_snapshot_path, weights_only=False):
            buffer.write_episode(ep)
        print(f'[resume] restored buffer: {buffer.num_steps_stored} steps, {buffer.num_episodes} episodes')

    probe_path = out_dir / 'eval_probe_buffer.pt'
    pairs_path = out_dir / 'eval_pairs.pt'
    if start_gen > 0 and probe_path.exists() and pairs_path.exists():
        probe_buffer = _TorchPixelReplayBuffer(
            max_steps=cfg.eval.num_probe_episodes * cfg.eval.probe_max_steps,
            history_len=cfg.history_size + cfg.num_preds,
            frameskip=1,
        )
        for ep in torch.load(probe_path, weights_only=False):
            probe_buffer.write_episode(ep)
        eval_pairs = torch.load(pairs_path, weights_only=False)
        print(f'[resume] restored fixed eval probe set ({len(eval_pairs)} pairs) from disk')
    else:
        probe_buffer, eval_pairs = build_eval_probe_set(cfg)
        # Fixed for the whole run (same probe set reused every generation) --
        # save once so the exact eval conditions are reproducible later
        # without re-running RandomPolicy collection (itself seeded, but
        # this avoids relying on that seeding matching bit-for-bit later).
        save_buffer_snapshot(probe_buffer, probe_path)
        torch.save(eval_pairs, pairs_path)
    for gen in range(start_gen, cfg.loop.n_generations):
        t0 = time.time()

        # -- 1. collect --------------------------------------------------
        # eval() is required (not just a formality): the planner rolls out
        # single-sample batches (one live env), and `module.MLP`'s
        # BatchNorm1d layers need running stats (not per-batch stats) to
        # handle batch size 1.
        model.eval()
        world.set_policy(explore_policy)
        world.collect(
            writer=buffer,
            episodes=cfg.loop.episodes_per_gen,
            seed=cfg.seed * 10_000 + gen,
        )
        collect_time = time.time() - t0

        # -- 2. train -----------------------------------------------------
        model.train()
        loader = torch.utils.data.DataLoader(
            buffer, batch_size=cfg.loop.batch_size, shuffle=True, drop_last=True
        )
        loader_iter = iter(loader)
        gen_logs = []
        for _ in range(cfg.loop.train_steps_per_gen):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            loss, logs, emb = lewm_train_step(
                model, sigreg, batch, cfg, ensemble=extras[0] if extras else None
            )
            loss.backward()
            optimizer.step()
            gen_logs.append(logs)
            global_step += 1

            if extras is not None and global_step % cfg.loop.density_update_every == 0:
                extras[1].update(emb.reshape(-1, emb.size(-1)))

        train_time = time.time() - t0 - collect_time
        avg_logs = {k: sum(l[k] for l in gen_logs) / len(gen_logs) for k in gen_logs[0]}

        # -- 3. eval --------------------------------------------------------
        eval_metrics = None
        if (gen + 1) % cfg.loop.eval_every == 0:
            model.eval()
            if eval_policy is None:
                eval_policy = make_eval_policy(cfg, model, device)
            eval_metrics = evaluate_on_probe_set(
                world, probe_buffer, eval_pairs, eval_policy, cfg
            )

        record = {
            'generation': gen,
            'global_step': global_step,
            'buffer_episodes': buffer.num_episodes,
            'buffer_steps': buffer.num_steps_stored,
            'collect_time_s': collect_time,
            'train_time_s': train_time,
            **avg_logs,
            **({f'eval/{k}': v for k, v in eval_metrics.items()} if eval_metrics else {}),
        }
        print(json.dumps(record))
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Data recording for future analysis without re-running: a
        # per-generation model+optimizer checkpoint (cheap -- tiny model)
        # and the *current* full training-buffer contents (pixels
        # included), overwritten each generation so a crash never loses
        # more than one generation's worth and disk use stays bounded to
        # ~one buffer's worth (buffer grows monotonically across
        # generations here, so this is the buffer's current, largest-so-far
        # size, not 8x that).
        save_checkpoint(
            ckpt_dir / f'gen_{gen:02d}.pt', gen, model, optimizer, extras, global_step
        )
        save_buffer_snapshot(buffer, out_dir / 'buffer_latest.pt')

    world.close()


if __name__ == '__main__':
    run()
