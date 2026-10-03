"""Can a latent world model order the candidates its planner actually chooses
between? This is the elite-set ranking-fidelity measurement behind Table 1
and Section 4.1 of the paper.

Motivation (Appendix E): auxiliary costs whose weights a calibration sweep
selects can contribute a tiny fraction of the goal cost's spread across
candidates, and still move success rate by several points. A term that
small cannot express a preference between candidates whose goal costs
differ appreciably. It can only decide among candidates the goal cost has
left nearly tied. So near-ties are outcome-relevant, which means the
elite-set boundary is being decided by cost differences at the very bottom
of the model's resolution.

This script measures that resolution directly. For a fixed initial state we take
the candidate set from the planner's final CEM iteration, and for each selected
candidate we compare

    J_model(c) = || zhat_H(c) - g ||^2     the cost the planner ranked on
    J_true(c)  = || z_H(c)    - g ||^2     the same cost, with the latent
                                           obtained by actually executing c

using the identical goal embedding g the model used (captured from the live
info_dict, so there is no chance of a mismatched normalization).

The headline quantities:

  rho_all     Spearman correlation over the whole sampled candidate set. Expected
              to be decent: telling good candidates from terrible ones is easy.
  rho_elite   Spearman correlation restricted to the top-k elite set. This is the
              ordering CEM actually refits on. If it is near zero the planner is
              refitting on noise.
  regret      true cost of the model's argmin, minus the best true cost
              available in the sampled set, in units of the true cost spread.
  k_eff       largest k such that the model's k-th ranked candidate is separated
              from its first by more than the model's own cost error.

Usage:
    python3 exp_rank_fidelity.py --env tworoom --n-episodes 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.stats import spearmanr

import stable_worldmodel as swm
from stable_worldmodel.planning.solver.callbacks import Callback
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal

import stable_worldmodel.planning.solver.cem as _cem_mod

_cem_mod.print = lambda *a, **k: None  # noqa: E731

from explore_train import _normalize_pixels, make_eval_policy  # noqa: E402
from pretrained import load_pretrained_model  # noqa: E402
from real_eval import build_real_eval  # noqa: E402

ENVS = ['tworoom', 'pusht', 'reacher', 'cube']


class GoalEmbCapture(nn.Module):
    """Transparent wrapper that keeps the last ``goal_emb`` the model was scored
    against. Using the live tensor rather than re-encoding the goal frame
    ourselves removes any risk that the true cost and the model cost are
    measured against different goal embeddings."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.goal_emb: torch.Tensor | None = None

    def forward(self, info_dict: dict) -> torch.Tensor:
        self.goal_emb = info_dict['goal_emb'].detach()
        return self.inner(info_dict)


class CandidateCapture(Callback):
    """Stores the candidate set and costs of the final CEM iteration of the
    first solve. One solve per episode is all we need, and the final iteration
    is the one whose elite set produces the executed action."""

    def __init__(self) -> None:
        super().__init__(reduction='none')
        self.last: dict[str, torch.Tensor] | None = None
        self.first_solve_done = False

    def reset(self) -> None:
        pass

    def compute(self, **state: Any) -> dict:
        if not self.first_solve_done:
            self.last = {
                'candidates': state['candidates'].detach()[0].cpu(),  # (N, H, A)
                'costs': state['costs'].detach()[0].cpu(),            # (N,)
            }
        return {}


def build_world(cfg, num_envs):
    env_kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True) if 'env_kwargs' in cfg else {}
    return swm.World(
        cfg.env_name, num_envs=num_envs,
        image_shape=(cfg.img_size, cfg.img_size),
        max_episode_steps=cfg.loop.max_episode_steps,
        **env_kwargs,
    )


def reset_batch_to(world, cfg, dataset, ep, start, m):
    """Put all m envs of `world` into the same dataset (ep, start) state.

    TwoRoom-style envs set state through `eval.callables`, which mutate the env
    without re-rendering, so `world.infos` is stale afterwards. We never read
    infos before the first step, so that is fine here.
    """
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [ep] * m, [start] * m, cfg.eval.goal_offset_steps
    )
    seeds = None
    if init_rows and 'seed' in init_rows[0]:
        seeds = [int(np.asarray(r['seed']).reshape(-1)[0]) for r in init_rows]
    has_method = hasattr(world.envs.envs[0].unwrapped, 'reset_options_from_dataset')
    if has_method:
        opts = [
            world.envs.envs[i].unwrapped.reset_options_from_dataset(init_rows[i], goal_rows[i])
            for i in range(m)
        ]
        world.reset(seed=seeds, options=opts)
    else:
        world.reset(seed=seeds)
        callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
        for i in range(m):
            _apply_callables(world.envs.envs[i].unwrapped, callables,
                             {**init_rows[i], **goal_rows[i]})
    return goal_rows


def rollout_candidates(world, cfg, dataset, ep, start, cands, process, model, device):
    """Execute each candidate open-loop from the same state.

    Returns `(terminal_latent, terminal_state)`. The latent is the reference that
    matches the planner's own cost exactly. The state is the same endpoint
    measured in the environment's coordinates, which removes the encoder from the
    comparison entirely: if the model's ranking fails against both, the failure
    cannot be blamed on the latent metric.
    """
    m = cands.shape[0]
    assert m == world.num_envs, (m, world.num_envs)
    goal_rows = reset_batch_to(world, cfg, dataset, ep, start, m)

    # (m, H, action_block * env_action_dim) -> (m, H * action_block, env_action_dim)
    env_action_dim = int(np.prod(world.envs.single_action_space.shape))
    steps = cands.reshape(m, -1, env_action_dim).numpy().astype(np.float32)

    px, last_infos = None, None
    for t in range(steps.shape[1]):
        a = steps[:, t].reshape(*world.envs.action_space.shape)
        if 'action' in process:
            a = process['action'].inverse_transform(a)
        _, _, _, _, infos = world.envs.step(a)
        world.infos = infos
        px = np.asarray(infos['pixels'])
        last_infos = infos

    frame = px[:, -1] if px.ndim > 4 else px
    t_px = torch.as_tensor(frame).to(device).float()
    if t_px.ndim == 4:
        t_px = t_px.unsqueeze(1)
    with torch.no_grad():
        emb = model.encode({'pixels': _normalize_pixels(t_px)})['emb']

    # state-space endpoint, in the env's own coordinates
    state_cost = None
    if 'proprio' in last_infos and goal_rows and 'goal_proprio' in goal_rows[0]:
        pro = np.asarray(last_infos['proprio'], dtype=np.float64)
        pro = pro[:, -1] if pro.ndim > 2 else pro          # (m, S)
        gp = np.stack([np.asarray(r['goal_proprio'], dtype=np.float64).reshape(-1)
                       for r in goal_rows])                 # (m, S)
        pro = pro.reshape(m, -1)
        if pro.shape[1] == gp.shape[1]:
            state_cost = ((pro - gp) ** 2).sum(axis=1)

    return emb[:, -1], state_cost  # (m, D), (m,) or None


def n_distinguishable(model_costs: np.ndarray, cost_err_std: float) -> int:
    """How many candidates the model can actually separate from the one it picked.

    A candidate counts as distinguishable when its model cost exceeds the best
    model cost by more than the model's own cost error. Zero means the model has
    no basis for preferring its pick over any other candidate in the set, so the
    elite set is one indistinguishable blob and which member becomes the next
    proposal mean is arbitrary.
    """
    best = model_costs.min()
    return int(((model_costs - best) > cost_err_std).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=ENVS, required=True)
    ap.add_argument('--n-episodes', type=int, default=50)
    ap.add_argument('--n-elite', type=int, default=30, help='how many top-ranked candidates to execute')
    ap.add_argument('--n-other', type=int, default=30, help='how many non-elite candidates to execute')
    ap.add_argument('--batch-envs', type=int, default=20)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = OmegaConf.load(f'config/explore/{args.env}.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.model_config_path = str(Path('config/train/model/lewm.yaml').resolve())
    out_dir = Path(f'outputs/runs/{args.env}/exp6_rank_fidelity')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'rank_fidelity.json'

    action_dim = cfg.plan_config.action_block * cfg.get('env_action_dim', 2)
    model = load_pretrained_model(args.env, action_dim, args.device)
    model.eval()
    keys = OmegaConf.to_container(cfg.eval.dataset_keys_to_cache, resolve=True)
    dataset, process, episodes_idx, start_steps = build_real_eval(
        cfg, dataset_name=cfg.eval.dataset_name, keys_to_cache=keys, seed=cfg.seed
    )
    pairs = list(zip(episodes_idx, start_steps))[: args.n_episodes]
    print(f'[{args.env}] {len(pairs)} episodes, topk={cfg.solver.topk}, '
          f'num_samples={cfg.solver.num_samples}', flush=True)

    plan_world = build_world(cfg, 1)
    roll_world = build_world(cfg, args.batch_envs)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    topk = int(cfg.solver.topk)

    # Validity check for the whole measurement: the true cost is only a
    # meaningful reference if re-executing the same candidate from the same
    # state gives the same latent. If the env is stochastic this number is
    # nonzero and every true cost below carries that much noise.
    determinism = None

    records = []
    for n, (ep, start) in enumerate(pairs):
        cap = CandidateCapture()
        goal_cap = GoalEmbCapture(swm.planning.GoalMSE())
        policy = make_eval_policy(cfg, model, args.device, process=process,
                                  objective=goal_cap, callbacks=[cap])
        plan_world.set_policy(policy)
        # one env step is enough: the policy replans on the first call, which is
        # the solve whose candidate set we want
        with torch.no_grad():
            plan_world.evaluate(
                dataset=dataset, episodes_idx=[ep], start_steps=[start],
                goal_offset=cfg.eval.goal_offset_steps, eval_budget=1,
                callables=callables,
            )
        if cap.last is None or goal_cap.goal_emb is None:
            print(f'[{args.env}] episode {ep}: no candidates captured, skipping', flush=True)
            continue

        cands = cap.last['candidates']          # (N, H, A)
        mcosts = cap.last['costs'].numpy()      # (N,)
        g = goal_cap.goal_emb[0, -1].to(args.device)  # (D,)

        if determinism is None:
            rep = cands[(order_top := np.argsort(mcosts)[:1])].repeat(args.batch_envs, 1, 1)
            l1, _ = rollout_candidates(roll_world, cfg, dataset, ep, start, rep,
                                       process, model, args.device)
            l2, _ = rollout_candidates(roll_world, cfg, dataset, ep, start, rep,
                                       process, model, args.device)
            c1 = (l1 - g[None, :]).pow(2).sum(-1)
            c2 = (l2 - g[None, :]).pow(2).sum(-1)
            determinism = {
                'within_batch_max_dev': float((c1 - c1[0]).abs().max()),
                'across_replay_max_dev': float((c1 - c2).abs().max()),
                'cost_scale': float(c1.mean()),
            }
            print(f'[{args.env}] determinism check: same candidate x{args.batch_envs} '
                  f"within-batch dev={determinism['within_batch_max_dev']:.3g}, "
                  f"re-executed dev={determinism['across_replay_max_dev']:.3g}, "
                  f"cost scale={determinism['cost_scale']:.3g}", flush=True)
            del order_top

        order = np.argsort(mcosts)
        elite_idx = order[: min(args.n_elite, topk)]
        rest = order[topk:]
        if len(rest) > 0 and args.n_other > 0:
            sel = np.linspace(0, len(rest) - 1, min(args.n_other, len(rest))).astype(int)
            other_idx = rest[sel]
        else:
            other_idx = np.array([], dtype=int)
        chosen = np.concatenate([elite_idx, other_idx])

        # execute in batches of exactly batch_envs (pad by repeating the last)
        true_lat, true_state = [], []
        for s in range(0, len(chosen), args.batch_envs):
            block = chosen[s : s + args.batch_envs]
            pad = args.batch_envs - len(block)
            block_p = np.concatenate([block, np.repeat(block[-1], pad)]) if pad else block
            lat, st = rollout_candidates(roll_world, cfg, dataset, ep, start,
                                         cands[block_p], process, model, args.device)
            true_lat.append(lat[: len(block)])
            if st is not None:
                true_state.append(st[: len(block)])
        true_lat = torch.cat(true_lat, dim=0)                     # (M, D)
        tcosts = (true_lat - g[None, :]).pow(2).sum(-1).cpu().numpy()
        scosts = np.concatenate(true_state) if len(true_state) == len(
            range(0, len(chosen), args.batch_envs)) else None
        msel = mcosts[chosen]

        n_el = len(elite_idx)
        rho_all = spearmanr(msel, tcosts).statistic if len(msel) > 2 else np.nan
        rho_el = spearmanr(msel[:n_el], tcosts[:n_el]).statistic if n_el > 2 else np.nan

        # Two error scales. The global one is measured over the whole sampled
        # range and is inflated by any scale mismatch across that range; the
        # elite-local one is the scale that actually matters for the elite
        # ordering, so we report both and rely on the local one.
        err_all = float(np.std(msel - tcosts))
        err_el = float(np.std(msel[:n_el] - tcosts[:n_el]))
        elite_range = float(msel[:n_el].max() - msel[:n_el].min())
        ndist = n_distinguishable(msel[:n_el], err_el)
        tspread = float(np.std(tcosts)) + 1e-12
        regret = float((tcosts[int(np.argmin(msel))] - tcosts.min()) / tspread)

        # the same two correlations against the env's own state space, which the
        # model's encoder plays no part in
        rho_all_s = rho_el_s = float('nan')
        if scosts is not None and np.std(scosts) > 0:
            rho_all_s = spearmanr(msel, scosts).statistic
            if n_el > 2 and np.std(scosts[:n_el]) > 0:
                rho_el_s = spearmanr(msel[:n_el], scosts[:n_el]).statistic

        rec = {
            'episode_idx': int(ep), 'start_step': int(start),
            'n_elite': int(n_el), 'n_total': int(len(chosen)),
            'rho_all': float(rho_all), 'rho_elite': float(rho_el),
            'n_distinguishable': ndist, 'regret_norm': regret,
            'cost_err_std_all': err_all, 'cost_err_std_elite': err_el,
            'model_cost_elite_range': elite_range,
            'elite_range_over_err': elite_range / (err_el + 1e-12),
            'rho_all_state': float(rho_all_s), 'rho_elite_state': float(rho_el_s),
            'model_costs': msel.tolist(), 'true_costs': tcosts.tolist(),
            'state_costs': scosts.tolist() if scosts is not None else None,
        }
        records.append(rec)
        print(f'[{args.env}] ep {n + 1}/{len(pairs)} rho_all={rho_all:+.3f} '
              f'rho_elite={rho_el:+.3f} | state-space rho_all={rho_all_s:+.3f} '
              f'rho_elite={rho_el_s:+.3f} | n_dist={ndist}/{n_el} '
              f'regret={regret:.3f}', flush=True)

        with open(out_path, 'w') as f:
            json.dump({'env': args.env, 'topk': topk,
                       'num_samples': int(cfg.solver.num_samples),
                       'determinism': determinism,
                       'records': records}, f, indent=2)

    plan_world.close()
    roll_world.close()

    if records:
        ra = np.array([r['rho_all'] for r in records])
        re_ = np.array([r['rho_elite'] for r in records])
        print(f'\n[{args.env}] SUMMARY over {len(records)} episodes')
        print(f'  rho_all   mean={np.nanmean(ra):+.3f} median={np.nanmedian(ra):+.3f}')
        print(f'  rho_elite mean={np.nanmean(re_):+.3f} median={np.nanmedian(re_):+.3f}')
        nd = np.array([r['n_distinguishable'] for r in records])
        print(f'  n_distinguishable mean={nd.mean():.2f} of {records[0]["n_elite"]} '
              f'(episodes with zero: {int((nd == 0).sum())}/{len(records)})')
        er = np.array([r['elite_range_over_err'] for r in records])
        print(f'  elite cost range / model cost error: median={np.median(er):.4f}')
        print(f'  regret    mean={np.mean([r["regret_norm"] for r in records]):.3f}')
        sa = np.array([r['rho_all_state'] for r in records], dtype=float)
        se = np.array([r['rho_elite_state'] for r in records], dtype=float)
        if not np.all(np.isnan(sa)):
            print(f'  state space (no encoder): rho_all={np.nanmean(sa):+.3f} '
                  f'rho_elite={np.nanmean(se):+.3f}')
    print(f'[{args.env}] saved -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
