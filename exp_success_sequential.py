"""Run a fresh, self-consistent sequential batch of episodes under the
terminal-frame baseline and the resolution-matched fix (one continuous
policy per arm, exactly the run_episodes() protocol used by every SR
number in this paper), saving one mp4 per episode per arm. Used to pick
a real illustrative case where the baseline fails and the fix succeeds,
for the qualitative comparison figure.

    python -u exp_success_sequential.py --env tworoom --frame 4 --n 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

import stable_worldmodel as swm
import stable_worldmodel.planning.solver.cem as _cem_mod

_cem_mod.print = lambda *a, **k: None  # noqa: E731

from exp_generality import build_world
from explore_train import make_eval_policy
from methods_original import FrameGoalMSE
from pretrained import load_pretrained_model
from real_eval import build_real_eval


def run_arm(world, pol, dataset, eval_pairs, cfg, video_root):
    world.set_policy(pol)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    succ = []
    for i, (ep, start) in enumerate(eval_pairs):
        result = world.evaluate(
            dataset=dataset, episodes_idx=[ep], start_steps=[start],
            goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
            callables=callables, video=str(video_root / f'ep_{i}'),
        )
        ok = bool(result['episode_successes'][0])
        succ.append(ok)
        print(f'  ep{i:3d} success={ok}', flush=True)
    return succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    cfg.loop.eval_episodes = 200

    sweep = json.loads(Path(f'outputs/runs/{args.env}/exp15_frame_sweep/sweep.json').read_text())
    lam_last = sweep['lams']['last']
    lam_frame = sweep['lams'][f'frame_{args.frame}']

    action_dim = cfg.plan_config.action_block * cfg.get('env_action_dim', 2)
    model = load_pretrained_model(args.env, action_dim, args.device)
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    eval_pairs = list(zip(episodes_idx, start_steps))[: args.n]

    out_root = Path(f'outputs/runs/{args.env}/exp21_sequential')

    print(f'[{args.env}] baseline_last, n={args.n}')
    world = build_world(cfg)
    objective = swm.planning.WeightedSum([(1.0, swm.planning.GoalMSE()), (lam_last, swm.planning.ControlPenalty())])
    pol = make_eval_policy(cfg, model, args.device, process=process, objective=objective)
    succ_last = run_arm(world, pol, dataset, eval_pairs, cfg, out_root / 'baseline_last')
    world.close()

    print(f'[{args.env}] fix_frame{args.frame}, n={args.n}')
    world = build_world(cfg)
    objective = swm.planning.WeightedSum([(1.0, FrameGoalMSE(args.frame)), (lam_frame, swm.planning.ControlPenalty())])
    pol = make_eval_policy(cfg, model, args.device, process=process, objective=objective)
    succ_frame = run_arm(world, pol, dataset, eval_pairs, cfg, out_root / 'fix_frame')
    world.close()

    flips = [i for i in range(args.n) if (not succ_last[i]) and succ_frame[i]]
    reverse = [i for i in range(args.n) if succ_last[i] and not succ_frame[i]]
    print(f'[{args.env}] baseline SR={100*sum(succ_last)/args.n:.1f}%  fix SR={100*sum(succ_frame)/args.n:.1f}%')
    print(f'[{args.env}] flips (baseline fail, fix succeed): {flips}')
    print(f'[{args.env}] reverse (baseline succeed, fix fail): {reverse}')
    (out_root / 'result.json').write_text(json.dumps({
        'succ_last': succ_last, 'succ_frame': succ_frame, 'flips': flips, 'reverse': reverse,
    }, indent=2))


if __name__ == '__main__':
    main()
