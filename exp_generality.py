"""Generality of the matched random-cost control (background for Appendix E).

A magnitude-matched random cost can reproduce a large share of the
success-rate gain that adding a model-derived auxiliary cost to a CEM cost
produces. That observation is about one specific auxiliary cost. This
script asks whether the same happens for auxiliary costs we did not
invent, which is what turns the observation into a statement about the
method class (any refitting search procedure) rather than about one
particular signal.

Three auxiliary costs share one harness:

  implausibility  GaussianNormPenalty, a model-derived risk signal
  control_penalty swm.planning.ControlPenalty, the L2-on-actions term that
                  appears in essentially every MPC paper
  rnd             RNDNovelty, Random Network Distillation (Burda et al. 2019)

Two design choices make the comparison fair, and both matter.

*Relative-strength grid.* The three costs live on wildly different scales, so a
shared grid of absolute lambdas would tune one of them well and the others
badly. Instead we calibrate the across-candidate spread of each cost and of
GoalMSE, then set lambda so that

    lambda * spread(aux) = r * spread(GoalMSE)

for r in a fixed grid. Every cost is then swept over the same range of *actual
influence on the ranking*, and nobody can say we tuned our own cost harder.

*One random control for the whole grid.* Because the control's noise is scaled
to match the spread of the cost it replaces, and lambda is chosen so that the
product equals r * spread(GoalMSE), the control at relative strength r is
numerically the same experiment whichever cost it stands in for. So the random
arm is a single curve over r, not three, and each real cost is read against it.

Spread, not mean, is the quantity matched: a term that is constant across
candidates cannot reorder them and so cannot change what the planner does.

Usage:
    python3 exp_generality.py --env tworoom
    python3 exp_generality.py --env cube --ratios 0.25 1.0 4.0 --n-eval 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import stable_worldmodel as swm

# The solvers print a timing line on every call, which at 200 episodes x ~30
# replans is tens of thousands of lines of log. Silence them at the module level
# before anything imports the classes.
import stable_worldmodel.planning.solver.cem as _cem_mod
import stable_worldmodel.planning.solver.mppi as _mppi_mod
import stable_worldmodel.planning.solver.predictive_sampling as _ps_mod

for _m in (_cem_mod, _mppi_mod, _ps_mod):
    _m.print = lambda *a, **k: None  # noqa: E731

from explore_train import make_eval_policy  # noqa: E402
from objectives_custom import GaussianNormPenalty  # noqa: E402
from objectives_generality import (  # noqa: E402
    MatchedRandomCost,
    RNDNovelty,
    SpreadRecorder,
    make_rnd_nets,
)
from pretrained import load_pretrained_model  # noqa: E402
from real_eval import build_real_eval  # noqa: E402

ENVS = ['tworoom', 'pusht', 'reacher', 'cube']
COSTS = ['implausibility', 'control_penalty', 'rnd']


def build_world(cfg):
    env_kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True) if 'env_kwargs' in cfg else {}
    return swm.World(
        cfg.env_name,
        num_envs=1,
        image_shape=(cfg.img_size, cfg.img_size),
        max_episode_steps=cfg.loop.max_episode_steps,
        **env_kwargs,
    )


def make_cost(name: str, cfg, env: str, device: str) -> torch.nn.Module:
    history_len = cfg.plan_config.get('history_len', 1)
    if name == 'implausibility':
        return GaussianNormPenalty(history_len=history_len)
    if name == 'control_penalty':
        return swm.planning.ControlPenalty()
    if name == 'rnd':
        ckpt = torch.load(f'rnd/{env}_rnd.pt', map_location=device, weights_only=False)
        target, predictor = make_rnd_nets(ckpt['emb_dim'], seed=ckpt['seed'])
        target.load_state_dict(ckpt['target'])
        predictor.load_state_dict(ckpt['predictor'])
        return RNDNovelty(target.to(device), predictor.to(device), history_len=history_len)
    raise ValueError(name)


def run_episodes(world, dataset, eval_pairs, eval_policy, cfg) -> list[bool]:
    world.set_policy(eval_policy)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    out = []
    with torch.no_grad():
        for ep, start in eval_pairs:
            result = world.evaluate(
                dataset=dataset,
                episodes_idx=[ep],
                start_steps=[start],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                callables=callables,
            )
            out.append(bool(result['episode_successes'][0]))
    return out


def sr(successes) -> float:
    return 100.0 * sum(successes) / max(len(successes), 1)


def run_arm(cfg, model, device, process, dataset, eval_pairs, extra, weight):
    """One evaluation arm: GoalMSE + weight * extra (extra=None -> baseline)."""
    world = build_world(cfg)
    if extra is None:
        objective = swm.planning.GoalMSE()
    else:
        objective = swm.planning.WeightedSum([(1.0, swm.planning.GoalMSE()), (weight, extra)])
    policy = make_eval_policy(cfg, model, device, process=process, objective=objective)
    succ = run_episodes(world, dataset, eval_pairs, policy, cfg)
    world.close()
    return succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--costs', nargs='+', choices=COSTS, default=COSTS)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.25, 1.0, 4.0],
                    help='auxiliary-term spread as a multiple of GoalMSE spread')
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--n-calib', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--calib-only', action='store_true',
                    help='measure spreads and exit, to choose the relative-strength grid')
    ap.add_argument('--paper-lambda', type=float, default=None,
                    help="implausibility lambda used in the paper, reported as a relative strength")
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = args.n_eval
    out_dir = Path(f'outputs/runs/{args.env}/exp5_generality')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'generality.json'

    action_dim = cfg.plan_config.action_block * cfg.get('env_action_dim', 2)
    model = load_pretrained_model(args.env, action_dim, args.device)
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    eval_pairs = list(zip(episodes_idx, start_steps))
    print(f'[{args.env}] {len(eval_pairs)} fixed (init, goal) pairs', flush=True)

    record: dict = {
        'env': args.env,
        'n_eval': len(eval_pairs),
        'ratios': args.ratios,
        'costs': args.costs,
    }

    def save():
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)

    # -- step 1: one calibration pass records the spread of GoalMSE and of every
    # auxiliary cost simultaneously, all at weight 0 so the planner is unchanged.
    goal_rec = SpreadRecorder(swm.planning.GoalMSE())
    aux_recs = {c: SpreadRecorder(make_cost(c, cfg, args.env, args.device)) for c in args.costs}
    terms = [(1.0, goal_rec)] + [(0.0, r) for r in aux_recs.values()]
    world = build_world(cfg)
    policy = make_eval_policy(
        cfg, model, args.device, process=process, objective=swm.planning.WeightedSum(terms)
    )
    run_episodes(world, dataset, eval_pairs[: args.n_calib], policy, cfg)
    world.close()

    goal_spread = goal_rec.summary()
    spreads = {c: r.summary() for c, r in aux_recs.items()}
    record['goal_spread'] = goal_spread
    record['aux_spreads'] = spreads
    print(f'[{args.env}] GoalMSE spread={goal_spread:.6g}', flush=True)
    for c, s in spreads.items():
        print(f'[{args.env}]   {c} spread={s:.6g}', flush=True)
    if goal_spread <= 0 or any(s <= 0 for s in spreads.values()):
        raise SystemExit('calibration produced a non-positive spread')

    # lambda that puts each cost at relative strength r
    record['lambdas'] = {
        c: {str(r): r * goal_spread / spreads[c] for r in args.ratios} for c in args.costs
    }
    if args.paper_lambda is not None and 'implausibility' in spreads:
        r_paper = args.paper_lambda * spreads['implausibility'] / goal_spread
        record['paper_lambda'] = args.paper_lambda
        record['paper_relative_strength'] = r_paper
        print(f'[{args.env}] paper lambda={args.paper_lambda} sits at relative strength '
              f'r={r_paper:.5g}', flush=True)
    save()
    if args.calib_only:
        print(f'[{args.env}] calibration only, saved -> {out_path}', flush=True)
        return

    # -- step 2: baseline, shared by every arm --
    baseline = run_arm(cfg, model, args.device, process, dataset, eval_pairs, None, 0.0)
    record['baseline_success'] = baseline
    record['baseline_sr'] = sr(baseline)
    print(f'[{args.env}] baseline sr={sr(baseline):.1f}%', flush=True)
    save()

    # -- step 3: the random arm, one curve over relative strength. Matched to
    # GoalMSE's spread directly, since lambda * spread(aux) = r * spread(goal)
    # makes the control identical whichever cost it replaces.
    record['random'] = {}
    for r in args.ratios:
        control = MatchedRandomCost(target_spread=goal_spread, seed=cfg.seed)
        succ = run_arm(cfg, model, args.device, process, dataset, eval_pairs, control, r)
        record['random'][str(r)] = {'success': succ, 'sr': sr(succ)}
        print(f'[{args.env}] random r={r} sr={sr(succ):.1f}% '
              f'(gain {sr(succ) - record["baseline_sr"]:+.1f})', flush=True)
        save()

    # -- step 4: each real cost over the same grid --
    record['real'] = {}
    for c in args.costs:
        record['real'][c] = {}
        for r in args.ratios:
            lam = record['lambdas'][c][str(r)]
            cost = make_cost(c, cfg, args.env, args.device)
            succ = run_arm(cfg, model, args.device, process, dataset, eval_pairs, cost, lam)
            record['real'][c][str(r)] = {'success': succ, 'sr': sr(succ), 'lambda': lam}
            print(f'[{args.env}] {c} r={r} (lam={lam:.4g}) sr={sr(succ):.1f}% '
                  f'(gain {sr(succ) - record["baseline_sr"]:+.1f})', flush=True)
            save()

    # -- summary: best-over-grid gain for each cost against the random arm --
    base = record['baseline_sr']
    best_random = max(record['random'][str(r)]['sr'] for r in args.ratios) - base
    record['best_random_gain'] = best_random
    record['summary'] = {}
    for c in args.costs:
        best_r = max(args.ratios, key=lambda r: record['real'][c][str(r)]['sr'])
        gain = record['real'][c][str(best_r)]['sr'] - base
        matched = record['random'][str(best_r)]['sr'] - base
        record['summary'][c] = {
            'best_ratio': best_r,
            'real_gain': gain,
            'random_gain_same_ratio': matched,
            'random_gain_best': best_random,
            'recovered_fraction_same_ratio': (matched / gain) if gain else None,
            'recovered_fraction_best': (best_random / gain) if gain else None,
        }
        print(f'[{args.env}] SUMMARY {c}: best r={best_r} real={gain:+.1f} '
              f'random@same_r={matched:+.1f} random@best={best_random:+.1f}', flush=True)
    save()
    print(f'[{args.env}] saved -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
