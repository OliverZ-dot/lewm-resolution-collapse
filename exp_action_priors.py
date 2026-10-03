"""Direction 3: how much of the benefit needs the world model at all?

Direction 1 established that the model cannot order the candidates inside its own
elite set: within that set its cost ranks pairs correctly about 51% of the time
against a 50% chance baseline. If that is the true state of affairs, then a term
that reads the *action sequence* -- and therefore needs no model, no latent, and
no uncertainty estimate -- should do at least as well as any term derived from the
model's predictions. This script supplies the model-free arms.

Two kinds of action-space prior are tested, because they enter the planner in
genuinely different places:

  a cost on the action sequence   `smoothness`, penalizing successive differences,
                                  swept over the same relative-strength grid as
                                  the model-derived costs so the comparison is
                                  like-for-like;
  a prior on the proposal        `colored noise` (iCEM, Pinneri et al. 2020),
                                  which biases *sampling* toward temporally
                                  correlated sequences and adds no cost term at
                                  all, so its weight cannot be tuned in our
                                  favour.

Everything else -- the baseline, the spread-matched random cost, and the
implausibility / control-penalty / RND arms -- is read from the Experiment 5
results rather than re-run, since the protocol, episode set and episode count are
identical by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import stable_worldmodel as swm

from exp_generality import ENVS, build_world, run_episodes, sr
from explore_train import make_eval_policy
from objectives_generality import SpreadRecorder
from pretrained import load_pretrained_model
from priors_action_space import ActionSmoothness, colored_noise_cem
from real_eval import build_real_eval


def run_colored(args, cfg, model, process, dataset, eval_pairs, record, save,
                base, action_dim, env_action_dim):
    """The colored-noise proposal prior: no cost term, so no weight to tune."""
    record['colored_noise'] = {}
    for beta in args.betas:
        world = build_world(cfg)
        policy = make_eval_policy(cfg, model, args.device, process=process,
                                  objective=swm.planning.GoalMSE())
        with colored_noise_cem(beta, int(cfg.solver.num_samples),
                               int(cfg.plan_config.horizon), action_dim,
                               env_action_dim) as hits:
            succ = run_episodes(world, dataset, eval_pairs, policy, cfg)
            n_hits = hits()
        world.close()
        # the shim is shape-guarded, so a zero count would mean it never fired
        # and we would silently be reporting a duplicate of the baseline
        assert n_hits > 0, f'colored-noise shim never fired for beta={beta}'
        record['colored_noise'][str(beta)] = {
            'success': succ, 'sr': sr(succ), 'shim_calls': n_hits}
        print(f'[{args.env}] colored noise beta={beta:g} SR={sr(succ):.1f}%  '
              f'(baseline {sr(base):.1f}%, {n_hits} candidate draws shaped)', flush=True)
        save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--ratios', type=float, nargs='+',
                    default=[0.0001, 0.001, 0.01, 0.1],
                    help='smoothness spread as a multiple of the GoalMSE spread')
    ap.add_argument('--betas', type=float, nargs='+', default=[1.0, 2.0],
                    help='colored-noise exponents; 0 would be the white-noise baseline')
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--n-calib', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--colored-only', action='store_true',
                    help='re-run just the colored-noise arms, keeping the saved '
                         'calibration, baseline and smoothness sweep')
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = args.n_eval
    out_dir = Path(f'outputs/runs/{args.env}/exp7_action_priors')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'action_priors.json'

    env_action_dim = int(cfg.get('env_action_dim', 2))
    action_dim = cfg.plan_config.action_block * env_action_dim
    model = load_pretrained_model(args.env, action_dim, args.device)
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    eval_pairs = list(zip(episodes_idx, start_steps))
    print(f'[{args.env}] {len(eval_pairs)} fixed (init, goal) pairs', flush=True)

    record: dict = {'env': args.env, 'n_eval': len(eval_pairs),
                    'ratios': args.ratios, 'betas': args.betas}
    if args.colored_only:
        if not out_path.exists():
            raise SystemExit(f'--colored-only needs an existing {out_path}')
        record = json.load(open(out_path))
        record['betas'] = args.betas

    def save():
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)

    if args.colored_only:
        base = record['baseline_success']
        print(f"[{args.env}] reusing saved baseline SR={record['baseline_sr']:.1f}%",
              flush=True)
        run_colored(args, cfg, model, process, dataset, eval_pairs, record, save,
                    base, action_dim, env_action_dim)
        print(f'[{args.env}] saved -> {out_path}', flush=True)
        return

    # -- calibration: measure the GoalMSE and smoothness spreads in one pass,
    # with the smoothness term at weight 0 so the planner is unchanged.
    goal_rec = SpreadRecorder(swm.planning.GoalMSE())
    sm_rec = SpreadRecorder(ActionSmoothness(env_action_dim))
    world = build_world(cfg)
    policy = make_eval_policy(
        cfg, model, args.device, process=process,
        objective=swm.planning.WeightedSum([(1.0, goal_rec), (0.0, sm_rec)]),
    )
    run_episodes(world, dataset, eval_pairs[: args.n_calib], policy, cfg)
    world.close()
    goal_spread, sm_spread = goal_rec.summary(), sm_rec.summary()
    record['goal_spread'] = goal_spread
    record['smoothness_spread'] = sm_spread
    print(f'[{args.env}] GoalMSE spread={goal_spread:.6g}  '
          f'smoothness spread={sm_spread:.6g}', flush=True)
    if goal_spread <= 0 or sm_spread <= 0:
        raise SystemExit('calibration produced a non-positive spread')
    record['lambdas'] = {str(r): r * goal_spread / sm_spread for r in args.ratios}
    save()

    # -- baseline, re-run here so the colored-noise arms are compared against a
    # baseline collected in the same process (Exp 5's baseline is the same
    # protocol, and agreement between the two is itself a useful check).
    world = build_world(cfg)
    policy = make_eval_policy(cfg, model, args.device, process=process,
                              objective=swm.planning.GoalMSE())
    base = run_episodes(world, dataset, eval_pairs, policy, cfg)
    world.close()
    record['baseline_success'] = base
    record['baseline_sr'] = sr(base)
    print(f'[{args.env}] baseline SR={sr(base):.1f}%', flush=True)
    save()

    # -- smoothness cost, swept over the same relative strengths as Exp 5
    record['smoothness'] = {}
    for r in args.ratios:
        lam = r * goal_spread / sm_spread
        world = build_world(cfg)
        obj = swm.planning.WeightedSum([
            (1.0, swm.planning.GoalMSE()),
            (lam, ActionSmoothness(env_action_dim)),
        ])
        policy = make_eval_policy(cfg, model, args.device, process=process, objective=obj)
        succ = run_episodes(world, dataset, eval_pairs, policy, cfg)
        world.close()
        record['smoothness'][str(r)] = {'lambda': lam, 'success': succ, 'sr': sr(succ)}
        print(f'[{args.env}] smoothness r={r:g} (lambda={lam:.4g}) '
              f'SR={sr(succ):.1f}%  (baseline {sr(base):.1f}%)', flush=True)
        save()

    run_colored(args, cfg, model, process, dataset, eval_pairs, record, save,
                base, action_dim, env_action_dim)
    print(f'[{args.env}] saved -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
