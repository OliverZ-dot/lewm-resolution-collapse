"""Exp 16: does the resolution-collapse point move with the planning horizon?

Exp 15a shows elite rho falls monotonically with predicted-frame index at the
paper's default H=5. That is consistent with two different mechanisms:

  (a) rollout error accumulates with *env steps into the future*, so collapse
      happens at a roughly fixed number of executed steps regardless of H;
  (b) collapse is a property of "how deep into whatever horizon CEM is
      planning", so it should track a fixed *fraction* of H.

Stretching H and re-measuring the curve distinguishes them. If the frame at
which elite rho crosses ~0 stays at the same env-step count under H=8 as
under H=5, that supports (a). If the crossing point scales with H, that
supports (b). Either answer is a finding: this is the first causal
manipulation of the resolution curve rather than a correlational fact about
one default configuration.

Only overrides cfg.plan_config.horizon / receding_horizon; everything else
(pretrained checkpoint, action_block, eval pairs) is identical to Exp 15a.

    python -u exp_horizon_length.py --env tworoom --horizon 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

import stable_worldmodel as swm
import stable_worldmodel.planning.solver.cem as _cem_mod

_cem_mod.print = lambda *a, **k: None  # noqa: E731

from exp_horizon_resolution import PredCapture, rollout_at_marks  # noqa: E402
from exp_rank_fidelity import CandidateCapture, build_world  # noqa: E402
from explore_train import make_eval_policy  # noqa: E402
from pretrained import load_pretrained_model  # noqa: E402
from real_eval import build_real_eval  # noqa: E402

ENVS = ['tworoom', 'pusht', 'reacher', 'cube']


def rho(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--horizon', type=int, required=True)
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
    base_h = int(cfg.plan_config.horizon)
    cfg.plan_config.horizon = args.horizon
    cfg.plan_config.receding_horizon = args.horizon
    out = Path(f'outputs/runs/{args.env}/exp16_horizon_length/h{args.horizon}.json')
    out.parent.mkdir(parents=True, exist_ok=True)

    env_action_dim = int(cfg.get('env_action_dim', 2))
    action_dim = args.horizon * env_action_dim  # unused by encoder, kept for API parity
    action_block = int(cfg.plan_config.action_block)
    model = load_pretrained_model(args.env, cfg.plan_config.action_block * env_action_dim, args.device)
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
    print(f'[{args.env}] H={args.horizon} (base {base_h})  {len(pairs)} episodes', flush=True)

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
        pred_all = pred_cap.predicted_emb[0].cpu()  # (S, T, D)
        goal = pred_cap.goal_emb[0, -1].cpu()
        t_len = int(pred_all.size(1))
        if pred_cap.action_candidates is not None:
            horizon = int(pred_cap.action_candidates.shape[2])
        else:
            horizon = args.horizon
        ctx = max(0, t_len - horizon)
        frame_idxs = list(range(ctx, t_len))
        marks = {f't{i}': (i - ctx + 1) * action_block for i in frame_idxs}
        if layout is None:
            layout = {
                'pred_len': t_len, 'horizon': horizon, 'context': ctx,
                'frame_idxs': frame_idxs, 'marks': marks, 'action_block': action_block,
            }
            print(f'[{args.env}] {layout}', flush=True)

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

        true_at = {f't{i}': [] for i in frame_idxs}
        for s in range(0, len(chosen), args.batch_envs):
            block = chosen[s : s + args.batch_envs]
            pad = args.batch_envs - len(block)
            block_p = np.concatenate([block, np.repeat(block[-1], pad)]) if pad else block
            embs = rollout_at_marks(
                roll_world, cfg, dataset, ep, start,
                cands[block_p], process, model, args.device, marks,
            )
            g = goal.to(args.device)
            for i in frame_idxs:
                key = f't{i}'
                lat = embs[key][: len(block)]
                true_at[key].append((lat - g[None]).pow(2).sum(-1).cpu().numpy())
        true_at = {k: np.concatenate(v) for k, v in true_at.items()}

        rec = {'episode_idx': int(ep), 'n_elite': int(n_el), 'n_total': int(len(chosen)), 'frames': {}}
        bits = []
        for i in frame_idxs:
            key = f't{i}'
            j = (pred_all[chosen, i] - goal).pow(2).sum(-1).numpy()
            t_same = true_at[key]
            rec['frames'][key] = {
                'rho_same_all': rho(j, t_same),
                'rho_same_elite': rho(j[:n_el], t_same[:n_el]),
            }
            bits.append(f'{key}={rec["frames"][key]["rho_same_elite"]:+.2f}')
        recs.append(rec)
        print(f'[{args.env}] H={args.horizon} ep {n + 1}/{len(pairs)}  elite ' + ' '.join(bits), flush=True)
        with open(out, 'w') as f:
            json.dump({'env': args.env, 'horizon': args.horizon, 'n_episodes': len(recs),
                       'layout': layout, 'records': recs}, f, indent=2)

    plan_world.close()
    roll_world.close()

    print(f'\n[{args.env}] H={args.horizon} SUMMARY n={len(recs)}', flush=True)
    for i in (layout or {}).get('frame_idxs', []):
        xs = np.array([r['frames'][f't{i}']['rho_same_elite'] for r in recs], float)
        print(f'  t{i} ({layout["marks"][f"t{i}"]} steps): elite={np.nanmean(xs):+.3f}', flush=True)
    print(f'[{args.env}] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
