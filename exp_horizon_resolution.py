"""Shared helpers for measuring elite-set rank fidelity at a chosen predicted
frame (terminal, mid-horizon, or otherwise), reused by the resolution-curve
and horizon-manipulation experiments (Section 4 of the paper).

Captures the predicted embedding trajectory from the final CEM score, so
different candidate frames are computed on the same candidates the planner
ranked. Executes those candidates open-loop and encodes the corresponding
true frame for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.stats import spearmanr

import stable_worldmodel as swm
import stable_worldmodel.planning.solver.cem as _cem_mod

_cem_mod.print = lambda *a, **k: None  # noqa: E731

from exp_rank_fidelity import (  # noqa: E402
    CandidateCapture,
    build_world,
    reset_batch_to,
)
from explore_train import _normalize_pixels, make_eval_policy  # noqa: E402
from pretrained import load_pretrained_model  # noqa: E402
from real_eval import build_real_eval  # noqa: E402

ENVS = ['tworoom', 'pusht', 'reacher', 'cube']


class PredCapture(nn.Module):
    """Keeps the live predicted trajectory and goal from the last score call."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.goal_emb: torch.Tensor | None = None
        self.predicted_emb: torch.Tensor | None = None
        self.action_candidates: torch.Tensor | None = None

    def forward(self, info_dict: dict) -> torch.Tensor:
        self.goal_emb = info_dict['goal_emb'].detach()
        self.predicted_emb = info_dict['predicted_emb'].detach()
        acts = info_dict.get('action_candidates')
        self.action_candidates = None if acts is None else acts.detach()
        return self.inner(info_dict)


def rho(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).statistic)


def encode_pixels(px, model, device):
    frame = px[:, -1] if px.ndim > 4 else px
    t_px = torch.as_tensor(frame).to(device).float()
    if t_px.ndim == 4:
        t_px = t_px.unsqueeze(1)
    with torch.no_grad():
        return model.encode({'pixels': _normalize_pixels(t_px)})['emb'][:, -1]


def rollout_at_marks(world, cfg, dataset, ep, start, cands, process, model, device, marks):
    """Execute each candidate; encode after the named env-step counts.

    `marks` maps a name to a 1-indexed step count. Step 0 is not used: after
    TwoRoom-style callables the rendered `infos` can be stale.
    """
    m = cands.shape[0]
    assert m == world.num_envs, (m, world.num_envs)
    reset_batch_to(world, cfg, dataset, ep, start, m)

    env_action_dim = int(np.prod(world.envs.single_action_space.shape))
    steps = cands.reshape(m, -1, env_action_dim).numpy().astype(np.float32)
    wanted = {int(v): k for k, v in marks.items() if v > 0}
    out = {k: None for k in marks}

    for t in range(steps.shape[1]):
        a = steps[:, t].reshape(*world.envs.action_space.shape)
        if 'action' in process:
            a = process['action'].inverse_transform(a)
        _, _, _, _, infos = world.envs.step(a)
        world.infos = infos
        if (t + 1) in wanted:
            out[wanted[t + 1]] = encode_pixels(np.asarray(infos['pixels']), model, device)

    return out


def scores_from_pred(pred: torch.Tensor, goal: torch.Tensor) -> dict[str, np.ndarray]:
    """pred (S, T, D), goal (D,)."""
    g = goal.view(1, 1, -1)
    last = (pred[:, -1] - goal).pow(2).sum(-1)
    t = pred.size(1)
    mid_i = max(0, t // 2)
    mid = (pred[:, mid_i] - goal).pow(2).sum(-1)
    start = (pred[:, 0] - goal).pow(2).sum(-1)
    return {
        'last': last.cpu().numpy(),
        'mid': mid.cpu().numpy(),
        'start': start.cpu().numpy(),
        'triple': (start + mid + last).cpu().numpy(),
        'progress': (pred - g).pow(2).sum(dim=(1, 2)).cpu().numpy(),
        'mid_index': mid_i,
        'pred_len': t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--n-episodes', type=int, default=50)
    ap.add_argument('--n-elite', type=int, default=30)
    ap.add_argument('--n-other', type=int, default=30)
    ap.add_argument('--batch-envs', type=int, default=20)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = args.n_episodes
    out = Path(f'outputs/runs/{args.env}/exp13_horizon/horizon.json')
    out.parent.mkdir(parents=True, exist_ok=True)

    env_action_dim = int(cfg.get('env_action_dim', 2))
    action_dim = cfg.plan_config.action_block * env_action_dim
    action_block = int(cfg.plan_config.action_block)
    model = load_pretrained_model(args.env, action_dim, args.device)
    model.eval()
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    pairs = list(zip(episodes_idx, start_steps))[: args.n_episodes]
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    topk = int(cfg.solver.topk)

    plan_world = build_world(cfg, 1)
    roll_world = build_world(cfg, args.batch_envs)
    recs = []
    layout = None
    print(f'[{args.env}] {len(pairs)} episodes', flush=True)

    for n, (ep, start) in enumerate(pairs):
        cap = CandidateCapture()
        pred_cap = PredCapture(swm.planning.GoalMSE())
        policy = make_eval_policy(
            cfg, model, args.device, process=process,
            objective=pred_cap, callbacks=[cap],
        )
        plan_world.set_policy(policy)
        with torch.no_grad():
            plan_world.evaluate(
                dataset=dataset, episodes_idx=[ep], start_steps=[start],
                goal_offset=cfg.eval.goal_offset_steps, eval_budget=1,
                callables=callables,
            )
        if cap.last is None or pred_cap.predicted_emb is None or pred_cap.goal_emb is None:
            print(f'[{args.env}] episode {ep}: no capture, skip', flush=True)
            continue

        cands = cap.last['candidates']
        mcosts = cap.last['costs'].numpy()
        pred_all = pred_cap.predicted_emb[0].cpu()
        goal = pred_cap.goal_emb[0, -1].cpu()
        scored = scores_from_pred(pred_all, goal)
        if pred_cap.action_candidates is not None:
            horizon = int(pred_cap.action_candidates.shape[2])
        else:
            horizon = int(cfg.plan_config.horizon)
        ctx = int(scored['pred_len']) - horizon
        mid_i = int(scored['mid_index'])
        last_steps = horizon * action_block
        if mid_i >= ctx:
            mid_steps = (mid_i - ctx + 1) * action_block
        else:
            mid_steps = 0
        if layout is None:
            layout = {
                'pred_len': int(scored['pred_len']),
                'horizon': horizon,
                'context': ctx,
                'mid_index': mid_i,
                'mid_env_steps': mid_steps,
                'last_env_steps': last_steps,
                'start_std_first_ep': float(np.std(scored['start'])),
            }
            print(
                f'[{args.env}] pred T={layout["pred_len"]} ctx={ctx} H={horizon} '
                f'mid_idx={mid_i} mid_steps={mid_steps} last_steps={last_steps} '
                f'start_std={layout["start_std_first_ep"]:.4g}',
                flush=True,
            )

        order = np.argsort(mcosts)
        elite_idx = order[: min(args.n_elite, topk)]
        rest = order[topk:]
        if len(rest) > 0 and args.n_other > 0:
            sel = np.linspace(0, len(rest) - 1, min(args.n_other, len(rest))).astype(int)
            other_idx = rest[sel]
        else:
            other_idx = np.array([], dtype=int)
        chosen = np.concatenate([elite_idx, other_idx])
        n_el = len(elite_idx)

        marks = {'last': last_steps}
        if mid_steps > 0:
            marks['mid'] = mid_steps

        true_last, true_mid = [], []
        for s in range(0, len(chosen), args.batch_envs):
            block = chosen[s : s + args.batch_envs]
            pad = args.batch_envs - len(block)
            block_p = np.concatenate([block, np.repeat(block[-1], pad)]) if pad else block
            embs = rollout_at_marks(
                roll_world, cfg, dataset, ep, start,
                cands[block_p], process, model, args.device, marks,
            )
            g = goal.to(args.device)
            lat_l = embs['last'][: len(block)]
            true_last.append((lat_l - g[None]).pow(2).sum(-1).cpu().numpy())
            if embs.get('mid') is not None:
                lat_m = embs['mid'][: len(block)]
                true_mid.append((lat_m - g[None]).pow(2).sum(-1).cpu().numpy())
        t_last = np.concatenate(true_last)
        t_mid = np.concatenate(true_mid) if true_mid else None

        rec = {
            'episode_idx': int(ep),
            'n_elite': int(n_el),
            'n_total': int(len(chosen)),
            'start_std': float(np.std(scored['start'][chosen])),
        }
        for name in ('last', 'mid', 'start', 'triple', 'progress'):
            j = scored[name][chosen]
            rec[f'{name}_all'] = j.tolist()
            rec[f'rho_{name}_vs_true_last_all'] = rho(j, t_last)
            rec[f'rho_{name}_vs_true_last_elite'] = rho(j[:n_el], t_last[:n_el])
            if t_mid is not None:
                rec[f'rho_{name}_vs_true_mid_all'] = rho(j, t_mid)
                rec[f'rho_{name}_vs_true_mid_elite'] = rho(j[:n_el], t_mid[:n_el])
        rec['true_last'] = t_last.tolist()
        rec['true_mid'] = None if t_mid is None else t_mid.tolist()
        recs.append(rec)
        print(
            f'[{args.env}] ep {n + 1}/{len(pairs)}  '
            f'last_el={rec["rho_last_vs_true_last_elite"]:+.2f} '
            f'mid_el={rec["rho_mid_vs_true_last_elite"]:+.2f} '
            f'trip_el={rec["rho_triple_vs_true_last_elite"]:+.2f}  '
            f'last_all={rec["rho_last_vs_true_last_all"]:+.2f} '
            f'mid_all={rec["rho_mid_vs_true_last_all"]:+.2f}',
            flush=True,
        )
        with open(out, 'w') as f:
            json.dump({
                'env': args.env, 'n_episodes': len(recs),
                'layout': layout, 'records': recs,
            }, f, indent=2)

    plan_world.close()
    roll_world.close()

    def mean(key):
        xs = np.array([r[key] for r in recs if key in r], dtype=float)
        return float(np.nanmean(xs)) if len(xs) else float('nan')

    print(f'\n[{args.env}] SUMMARY n={len(recs)}', flush=True)
    for name in ('last', 'mid', 'triple', 'start', 'progress'):
        print(
            f'  {name:8s} vs true last | all={mean(f"rho_{name}_vs_true_last_all"):+.3f}  '
            f'elite={mean(f"rho_{name}_vs_true_last_elite"):+.3f}',
            flush=True,
        )
    if recs and recs[0].get('true_mid') is not None:
        print(
            f'  mid      vs true mid  | all={mean("rho_mid_vs_true_mid_all"):+.3f}  '
            f'elite={mean("rho_mid_vs_true_mid_elite"):+.3f}',
            flush=True,
        )
    print(f'[{args.env}] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
