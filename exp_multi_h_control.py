"""Exp 17: is the MultiHorizon (mid+last+L2) gain a random-noise confound?

Exp 5/14 showed implausibility's gain is indistinguishable from a
spread-matched random cost added to GoalMSE. MultiHorizon (Exp 12/14/15)
also works by adding a term to GoalMSE_last -- the mid-horizon GoalMSE --
before the certified L2. The same control has to be run on it, or the
"resolution-aware method" claim is exposed to the same objection the paper
already raises against implausibility.

Three arms, same certified-L2 methodology as exp_fair_horizon.py:

  last            GoalMSE_last + lam_l2 * L2                      (baseline)
  multi_h_l2      GoalMSE_last + MidOnly + lam_l2 * L2             (the method)
  randmid_l2      GoalMSE_last + MatchedRandomCost(spread=mid) + lam_l2 * L2
                                                                    (control)

`lam_l2` is fixed across all three arms (computed once from the calibrated
spread of GoalMSE_last + MidOnly, exactly as Exp 14's multi_h_l2), so the only
thing that differs between multi_h_l2 and randmid_l2 is whether the added
term is a real mid-horizon distance-to-goal or noise of the same spread.

If multi_h_l2 beats randmid_l2, the mid-horizon term's *content* -- not just
its magnitude -- is doing the work, which is a materially stronger claim than
anything implausibility survived. If it does not, MultiHorizon is exposed to
the same confound and the resolution-aware method needs to be walked back.

    python -u exp_multi_h_control.py --env reacher --n-eval 200
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
from methods_original import MidOnly, MultiHorizon
from objectives_generality import MatchedRandomCost, SpreadRecorder
from pretrained import load_pretrained_model
from real_eval import build_real_eval

ARMS = ['last', 'multi_h_l2', 'randmid_l2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--l2-r', type=float, default=0.1)
    ap.add_argument('--n-calib', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0, help='seed for the noise generator')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--arms', nargs='+', default=ARMS)
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = args.n_eval
    out = Path(f'outputs/runs/{args.env}/exp17_multi_h_control/control.json')
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
    print(f'[{args.env}] {len(eval_pairs)} pairs  seed={cfg.seed}  r={args.l2_r}', flush=True)

    # --- calibration: spread of GoalMSE_last, MidOnly, MultiHorizon (=last+mid), L2 ---
    recs = {
        'last': SpreadRecorder(swm.planning.GoalMSE()),
        'mid': SpreadRecorder(MidOnly()),
        'mh': SpreadRecorder(MultiHorizon()),
        'l2': SpreadRecorder(swm.planning.ControlPenalty()),
    }
    world = build_world(cfg)
    pol = make_eval_policy(
        cfg, model, args.device, process=process,
        objective=swm.planning.WeightedSum([(1.0, recs['last'])] + [(0.0, v) for k, v in recs.items() if k != 'last']),
    )
    run_episodes(world, dataset, eval_pairs[: args.n_calib], pol, cfg)
    world.close()

    spreads = {k: v.summary() for k, v in recs.items()}
    l2_spread = max(spreads['l2'], 1e-12)
    lam_l2 = args.l2_r * spreads['mh'] / l2_spread
    print(f'[{args.env}] spreads ' + ' '.join(f'{k}={v:.4g}' for k, v in spreads.items()), flush=True)
    print(f'[{args.env}] lam_l2={lam_l2:.4g}  random-control target spread(mid)={spreads["mid"]:.4g}', flush=True)

    record = {
        'env': args.env, 'n_eval': len(eval_pairs), 'seed': int(cfg.seed),
        'l2_r': args.l2_r, 'spreads': spreads, 'lam_l2': lam_l2,
        'noise_seed': args.seed, 'arms': {},
    }
    if out.exists():
        prev = json.load(open(out))
        if prev.get('n_eval') == args.n_eval and abs(float(prev.get('l2_r', -1)) - args.l2_r) < 1e-12:
            record['arms'] = prev.get('arms') or {}

    def save():
        with open(out, 'w') as f:
            json.dump(record, f, indent=2)

    mains = {
        'last': [(1.0, swm.planning.GoalMSE())],
        'multi_h_l2': [(1.0, swm.planning.GoalMSE()), (1.0, MidOnly())],
        'randmid_l2': [
            (1.0, swm.planning.GoalMSE()),
            (1.0, MatchedRandomCost(target_spread=spreads['mid'], seed=args.seed)),
        ],
    }

    for name in args.arms:
        if name not in mains:
            raise ValueError(name)
        done = record['arms'].get(name)
        if done and len(done.get('success') or []) >= args.n_eval:
            print(f'[{args.env}] {name:16s} already done SR={done["sr"]:.1f}%', flush=True)
            continue
        world = build_world(cfg)
        pol = make_eval_policy(
            cfg, model, args.device, process=process,
            objective=swm.planning.WeightedSum(mains[name] + [(lam_l2, swm.planning.ControlPenalty())]),
        )
        succ = run_episodes(world, dataset, eval_pairs, pol, cfg)
        world.close()
        record['arms'][name] = {'sr': sr(succ), 'success': succ, 'lam_l2': lam_l2}
        print(f'[{args.env}] {name:16s} SR={sr(succ):.1f}%', flush=True)
        save()

    print(f'[{args.env}] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
