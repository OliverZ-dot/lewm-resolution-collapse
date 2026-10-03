"""Exp 15b: planner success when GoalMSE is asked at each predicted frame.

Every arm is main-cost + action-L2 at the same relative strength r=0.1
used to certify L2 in Exp 5 / Exp 14. Main costs:

  frame_k     GoalMSE at predicted-embedding index k
  decaying    earlier frames weighted more (see DecayingPath)
  multi_h     mid + last  (replication of Exp 14)
  last        terminal GoalMSE  (the certified L2 baseline)

    python -u exp_frame_sweep.py --env reacher --n-eval 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

import stable_worldmodel as swm
import stable_worldmodel.planning.solver.cem as _cem_mod

_cem_mod.print = lambda *a, **k: None  # noqa: E731

from exp_generality import ENVS, build_world, run_episodes, sr
from explore_train import make_eval_policy
from methods_original import DecayingPath, FrameGoalMSE, MultiHorizon
from objectives_generality import SpreadRecorder
from pretrained import load_pretrained_model
from real_eval import build_real_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--l2-r', type=float, default=0.1)
    ap.add_argument('--n-calib', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--arms', nargs='+', default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = args.n_eval
    out = Path(f'outputs/runs/{args.env}/exp15_frame_sweep/sweep.json')
    out.parent.mkdir(parents=True, exist_ok=True)

    action_dim = cfg.plan_config.action_block * cfg.get('env_action_dim', 2)
    model = load_pretrained_model(args.env, action_dim, args.device)
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    eval_pairs = list(zip(episodes_idx, start_steps))
    if len(eval_pairs) != args.n_eval:
        raise RuntimeError(f'eval_pairs={len(eval_pairs)} != n_eval={args.n_eval}')

    # Predicted-emb layout is 1 context + H predictions. Indices 1..H.
    horizon = int(cfg.plan_config.horizon)
    frame_idxs = list(range(1, horizon + 1))
    default_arms = [f'frame_{i}' for i in frame_idxs] + ['decaying', 'multi_h', 'last']
    arms = args.arms or default_arms

    mains = {f'frame_{i}': FrameGoalMSE(i) for i in frame_idxs}
    mains['decaying'] = DecayingPath()
    mains['multi_h'] = MultiHorizon()
    mains['last'] = swm.planning.GoalMSE()

    print(f'[{args.env}] {len(eval_pairs)} pairs  seed={cfg.seed}  r={args.l2_r}  arms={arms}', flush=True)

    recs = {name: SpreadRecorder(mains[name]) for name in mains}
    recs['l2'] = SpreadRecorder(swm.planning.ControlPenalty())
    world = build_world(cfg)
    pol = make_eval_policy(
        cfg, model, args.device, process=process,
        objective=swm.planning.WeightedSum(
            [(1.0, recs['last']), (0.0, recs['l2'])] +
            [(0.0, recs[n]) for n in mains if n != 'last']
        ),
    )
    run_episodes(world, dataset, eval_pairs[: args.n_calib], pol, cfg)
    world.close()

    spreads = {k: v.summary() for k, v in recs.items()}
    l2_spread = max(spreads['l2'], 1e-12)

    def lam(main_key: str) -> float:
        return args.l2_r * max(spreads[main_key], 1e-12) / l2_spread

    lams = {name: lam(name) for name in mains}
    print(f'[{args.env}] spreads ' + ' '.join(f'{k}={v:.4g}' for k, v in spreads.items()), flush=True)
    print(f'[{args.env}] lam     ' + ' '.join(f'{k}={v:.4g}' for k, v in lams.items()), flush=True)

    record = {
        'env': args.env,
        'n_eval': len(eval_pairs),
        'seed': int(cfg.seed),
        'l2_r': args.l2_r,
        'frame_idxs': frame_idxs,
        'spreads': spreads,
        'lams': lams,
        'arms': {},
    }
    if out.exists():
        prev = json.load(open(out))
        if prev.get('n_eval') == args.n_eval and abs(float(prev.get('l2_r', -1)) - args.l2_r) < 1e-12:
            record['arms'] = prev.get('arms') or {}
            record['spreads'] = prev.get('spreads') or spreads
            record['lams'] = prev.get('lams') or lams

    def save():
        with open(out, 'w') as f:
            json.dump(record, f, indent=2)

    for name in arms:
        if name not in mains:
            raise ValueError(name)
        done = record['arms'].get(name)
        if done and len(done.get('success') or []) >= args.n_eval:
            print(f'[{args.env}] {name:16s} already done SR={done["sr"]:.1f}%', flush=True)
            continue
        world = build_world(cfg)
        pol = make_eval_policy(
            cfg, model, args.device, process=process,
            objective=swm.planning.WeightedSum([
                (1.0, mains[name]),
                (lams[name], swm.planning.ControlPenalty()),
            ]),
        )
        succ = run_episodes(world, dataset, eval_pairs, pol, cfg)
        world.close()
        record['arms'][name] = {
            'sr': sr(succ), 'success': succ, 'lam': lams[name], 'r': args.l2_r,
        }
        print(f'[{args.env}] {name:16s} SR={sr(succ):.1f}%  lam={lams[name]:.4g}', flush=True)
        save()

    print(f'[{args.env}] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
