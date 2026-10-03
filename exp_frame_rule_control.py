"""Exp 20: is the rule-selected frame's gain (Exp 19) a random-noise confound?

Exp 19 shows a resolution-curve-driven rule (frac=0.7 of rho(1)) picks a
single frame per environment (frame_4 on TwoRoom/Reacher/PushT, frame_3 on
Cube) that is statistically indistinguishable from the success-rate sweep's
oracle. But `FrameGoalMSE(frame*)` *replaces* GoalMSE_last as the entire main
cost (not an additive term on top of it, unlike MultiHorizon in Exp 17), so
the natural objection is sharper here: maybe CEM doesn't need a real
distance-to-goal signal at all, and a cost of the right *magnitude* combined
with the certified L2 prior would do just as well as the real frame_k.

Two arms, reusing Exp 15b's own spread/lambda calibration for frame* exactly
(no recalibration -- same lam, same spread target, so this is a drop-in
substitution of content for noise):

  frame_star        FrameGoalMSE(frame*) + lam * L2      (already in Exp 15b)
  rand_frame_star   MatchedRandomCost(spread=spread(frame_star)) + lam * L2

If `frame_star` beats `rand_frame_star`, the model's prediction at that
specific frame is carrying real information CEM can use to rank candidates,
not just adding search variance of the right size. If it doesn't, the
rule-selected frame is exposed to the same confound Exp 5 found for
implausibility, despite passing Exp 19's oracle-matching test.

    python -u exp_frame_rule_control.py --env reacher
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
from methods_original import FrameGoalMSE
from objectives_generality import MatchedRandomCost
from pretrained import load_pretrained_model
from real_eval import build_real_eval

# Exp 19 primary rule (frac=0.7): frame* per environment, read off the
# calibration curve, blind to success rate.
FRAME_STAR = {'tworoom': 4, 'reacher': 4, 'pusht': 4, 'cube': 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0, help='seed for the noise generator')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    frame_star = FRAME_STAR[args.env]
    sweep_path = Path(f'outputs/runs/{args.env}/exp15_frame_sweep/sweep.json')
    sweep = json.loads(sweep_path.read_text())
    key = f'frame_{frame_star}'
    lam = sweep['lams'][key]
    target_spread = sweep['spreads'][key]
    print(f'[{args.env}] frame*={frame_star}  lam={lam:.4g}  target_spread={target_spread:.4g}', flush=True)

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = args.n_eval
    out = Path(f'outputs/runs/{args.env}/exp20_frame_rule_control/control.json')
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
    if sweep.get('n_eval') != args.n_eval or sweep.get('seed') != int(cfg.seed):
        raise RuntimeError('sweep.json eval config does not match this run; re-check n_eval/seed')
    print(f'[{args.env}] {len(eval_pairs)} pairs  seed={cfg.seed}', flush=True)

    record = {
        'env': args.env, 'n_eval': len(eval_pairs), 'seed': int(cfg.seed),
        'frame_star': frame_star, 'lam': lam, 'target_spread': target_spread,
        'noise_seed': args.seed,
        'arms': {
            'last': sweep['arms']['last'],
            'frame_star': sweep['arms'][key],
        },
    }
    if out.exists():
        prev = json.load(open(out))
        if prev.get('n_eval') == args.n_eval and prev.get('frame_star') == frame_star:
            if 'rand_frame_star' in (prev.get('arms') or {}):
                record['arms']['rand_frame_star'] = prev['arms']['rand_frame_star']

    def save():
        with open(out, 'w') as f:
            json.dump(record, f, indent=2)

    if len(record['arms'].get('rand_frame_star', {}).get('success') or []) >= args.n_eval:
        print(f'[{args.env}] rand_frame_star already done SR={record["arms"]["rand_frame_star"]["sr"]:.1f}%', flush=True)
    else:
        world = build_world(cfg)
        pol = make_eval_policy(
            cfg, model, args.device, process=process,
            objective=swm.planning.WeightedSum([
                (1.0, MatchedRandomCost(target_spread=target_spread, seed=args.seed)),
                (lam, swm.planning.ControlPenalty()),
            ]),
        )
        succ = run_episodes(world, dataset, eval_pairs, pol, cfg)
        world.close()
        record['arms']['rand_frame_star'] = {'sr': sr(succ), 'success': succ, 'lam': lam}
        print(f'[{args.env}] rand_frame_star  SR={sr(succ):.1f}%', flush=True)
        save()

    last_sr = record['arms']['last']['sr']
    fs_sr = record['arms']['frame_star']['sr']
    rand_sr = record['arms']['rand_frame_star']['sr']
    print(f'[{args.env}] last={last_sr:.1f}%  frame_{frame_star}={fs_sr:.1f}%  '
          f'rand(matched)={rand_sr:.1f}%', flush=True)
    print(f'[{args.env}] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
