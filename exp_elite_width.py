"""Direction 1, the design consequence: does widening the elite set remove the
random-cost confound?

Direction 1 measured the world model's resolution over its own candidate set. Two
numbers matter here. Within the top-30 elite set the model orders pairs correctly
about 51% of the time against a 50% chance baseline, and the elite set's entire
cost range is between 6% and 30% of the model's own cost error. So the rank-30
boundary is not a distinction the model is able to make: it separates candidates
the model cannot tell apart, and the resulting proposal refit inherits whatever
noise decided the ordering.

That gives a falsifiable prediction. The elite fraction k/N controls how hard CEM
commits to an ordering it cannot resolve. Small k means an aggressive refit onto a
handful of candidates picked out by noise; large k averages the noise away and, in
the limit k = N, the refit stops depending on the ordering at all. Experiment 5
found that the gain from a spread-matched *random* cost disappears under a solver
that does not refit its proposal. If that gain is really the signature of
refitting on an unresolvable ordering, then it should shrink monotonically as k
grows, with no change to the model, the cost, or the environment.

The experiment sweeps k and, at each k, measures both the plain baseline and the
best a spread-matched random cost can do. Every k receives the *same* weight sweep
(all four relative strengths), which matters: the strengths reported in
Experiment 5 were selected at the default k = 30, and reusing that choice here
would hand k = 30 a tuning advantage and manufacture the predicted decay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

import stable_worldmodel as swm

from exp_generality import ENVS, build_world, run_episodes, sr
from explore_train import make_eval_policy
from objectives_generality import MatchedRandomCost, SpreadRecorder
from pretrained import load_pretrained_model
from real_eval import build_real_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--topks', type=int, nargs='+', default=[5, 15, 30, 60, 150],
                    help='elite-set sizes; the LeWM default is 30 of 300 candidates')
    ap.add_argument('--ratios', type=float, nargs='+',
                    default=[0.0001, 0.001, 0.01, 0.1],
                    help='random-cost spread as a multiple of the GoalMSE spread')
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--n-calib', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0, help='seed of the random cost')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = args.n_eval
    out_dir = Path(f'outputs/runs/{args.env}/exp8_elite_width')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'elite_width.json'

    action_dim = cfg.plan_config.action_block * cfg.get('env_action_dim', 2)
    model = load_pretrained_model(args.env, action_dim, args.device)
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    eval_pairs = list(zip(episodes_idx, start_steps))
    num_samples = int(cfg.solver.num_samples)
    print(f'[{args.env}] {len(eval_pairs)} pairs, num_samples={num_samples}, '
          f'default topk={cfg.solver.topk}', flush=True)

    record: dict = {'env': args.env, 'n_eval': len(eval_pairs),
                    'num_samples': num_samples, 'default_topk': int(cfg.solver.topk),
                    'topks': args.topks, 'ratios': args.ratios, 'by_topk': {}}

    def save():
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)

    for k in args.topks:
        cfg.solver.topk = int(k)
        entry: dict = {'elite_fraction': k / num_samples}

        # The candidate spread depends on the proposal, and the proposal depends
        # on k, so the random cost is recalibrated at every k rather than reusing
        # one global scale.
        goal_rec = SpreadRecorder(swm.planning.GoalMSE())
        world = build_world(cfg)
        policy = make_eval_policy(cfg, model, args.device, process=process,
                                  objective=goal_rec)
        run_episodes(world, dataset, eval_pairs[: args.n_calib], policy, cfg)
        world.close()
        goal_spread = goal_rec.summary()
        entry['goal_spread'] = goal_spread
        if goal_spread <= 0:
            raise SystemExit(f'non-positive GoalMSE spread at topk={k}')

        world = build_world(cfg)
        policy = make_eval_policy(cfg, model, args.device, process=process,
                                  objective=swm.planning.GoalMSE())
        base = run_episodes(world, dataset, eval_pairs, policy, cfg)
        world.close()
        entry['baseline_success'] = base
        entry['baseline_sr'] = sr(base)
        print(f'[{args.env}] topk={k:3d} (elite frac {k / num_samples:.3f}) '
              f'baseline SR={sr(base):.1f}%  goal_spread={goal_spread:.4g}', flush=True)

        entry['random'] = {}
        for r in args.ratios:
            world = build_world(cfg)
            obj = swm.planning.WeightedSum([
                (1.0, swm.planning.GoalMSE()),
                (1.0, MatchedRandomCost(target_spread=r * goal_spread, seed=args.seed)),
            ])
            policy = make_eval_policy(cfg, model, args.device, process=process,
                                      objective=obj)
            succ = run_episodes(world, dataset, eval_pairs, policy, cfg)
            world.close()
            entry['random'][str(r)] = {'sr': sr(succ), 'success': succ,
                                       'target_spread': r * goal_spread}
            print(f'[{args.env}]   topk={k:3d} random r={r:g} SR={sr(succ):.1f}% '
                  f'(gain {sr(succ) - sr(base):+.1f})', flush=True)
            record['by_topk'][str(k)] = entry
            save()

        gains = {r: entry['random'][r]['sr'] - entry['baseline_sr'] for r in entry['random']}
        best_r = max(gains, key=gains.get)
        entry['best_random_gain'] = gains[best_r]
        entry['best_ratio'] = best_r
        print(f'[{args.env}] topk={k:3d} best random gain = {gains[best_r]:+.1f} '
              f'at r={best_r}', flush=True)
        record['by_topk'][str(k)] = entry
        save()

    print(f'[{args.env}] saved -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
