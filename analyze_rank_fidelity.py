"""Analysis for the candidate-ranking-fidelity experiment (Direction 1).

Reads `outputs/runs/<env>/exp6_rank_fidelity/rank_fidelity.json` and answers, for
each environment, whether the world model can order the candidates its planner
actually chooses between.

The headline contrast is rho_all vs rho_elite: the model's cost ordering agrees
with the truth over the whole candidate range but not inside the elite set, which
is the only ordering CEM uses. Three objections have to be closed before that
contrast means anything, and each gets its own measurement here.

1. Restriction of range. If the elite candidates all led to nearly the same true
   outcome, failing to order them would be harmless rather than a failure. We
   report the elite set's true-cost spread as a fraction of the full sampled
   set's, plus the regret actually incurred, to show the outcomes differ.

2. Noise in the reference. The true cost is only exact if re-executing a
   candidate reproduces its latent. The experiment records that deviation; here
   we turn it into an upper bound on the correlation any predictor could reach,
   so an attenuated rho_elite is not mistaken for a model failure.

3. Scale and monotonicity artifacts in Spearman. We also compute pairwise
   ranking accuracy as a function of the model's own predicted cost gap. That is
   scale-free, needs no assumption about the cost's functional form, and is
   stated in the units the planner acts on: when the model believes A beats B by
   delta, how often is it right?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

ENVS = ['tworoom', 'pusht', 'reacher', 'cube']
LABEL = {'tworoom': 'TwoRoom', 'pusht': 'PushT', 'reacher': 'Reacher', 'cube': 'Cube'}
FIG_DIR = Path('figures/direction1')


def load(env):
    p = Path(f'outputs/runs/{env}/exp6_rank_fidelity/rank_fidelity.json')
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def pairwise_accuracy(records, elite_only, n_bins=8):
    """Pooled over episodes: P(model orders a pair correctly) vs the model's gap.

    Pairs are formed within an episode only -- costs are not comparable across
    initial states. Returns (bin centers, accuracy, count, gap values) with the
    gap axis in units of that episode's elite-set model-cost range, so episodes
    with different cost scales can be pooled.
    """
    gaps, correct = [], []
    for r in records:
        m = np.asarray(r['model_costs'])
        t = np.asarray(r['true_costs'])
        n_el = r['n_elite']
        if elite_only:
            m, t = m[:n_el], t[:n_el]
        if len(m) < 3:
            continue
        scale = float(np.asarray(r['model_costs'])[:n_el].std()) + 1e-12
        i, j = np.triu_indices(len(m), k=1)
        dm, dt = m[i] - m[j], t[i] - t[j]
        keep = (dm != 0) & (dt != 0)
        gaps.append(np.abs(dm[keep]) / scale)
        correct.append((np.sign(dm[keep]) == np.sign(dt[keep])).astype(float))
    if not gaps:
        return None
    gaps = np.concatenate(gaps)
    correct = np.concatenate(correct)
    qs = np.quantile(gaps, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    centers, accs, counts = [], [], []
    for a, b in zip(qs[:-1], qs[1:]):
        sel = (gaps >= a) & (gaps < b)
        if sel.sum() < 30:
            continue
        centers.append(float(np.median(gaps[sel])))
        accs.append(float(correct[sel].mean()))
        counts.append(int(sel.sum()))
    return np.array(centers), np.array(accs), np.array(counts), gaps


def attenuation_bound(data, records):
    """Max Spearman any predictor could reach given noise in the true cost.

    The determinism check re-executes one candidate and reports the spread of its
    true cost. Treating that as independent measurement noise of std s_ref on a
    signal whose elite spread is s_sig, the reliability of the reference is
    s_sig^2 / (s_sig^2 + s_ref^2) and the attainable correlation is its sqrt.
    """
    det = data.get('determinism') or {}
    s_ref = max(float(det.get('within_batch_max_dev', 0.0)),
                float(det.get('across_replay_max_dev', 0.0))) / 2.0
    sig = np.mean([np.std(np.asarray(r['true_costs'])[: r['n_elite']]) for r in records])
    if sig <= 0:
        return np.nan, s_ref, sig
    rel = sig**2 / (sig**2 + s_ref**2)
    return float(np.sqrt(rel)), s_ref, float(sig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows, pw_elite, pw_all, per_env = [], {}, {}, {}

    for env in ENVS:
        data = load(env)
        if data is None:
            print(f'[{env}] missing, skipped')
            continue
        recs = data['records']
        ra = np.array([r['rho_all'] for r in recs], dtype=float)
        re_ = np.array([r['rho_elite'] for r in recs], dtype=float)
        ok = ~(np.isnan(ra) | np.isnan(re_))
        ra, re_ = ra[ok], re_[ok]

        # A failed test against zero would not establish that the elite ordering
        # is uninformative, so we report a bootstrap upper confidence bound on
        # rho_elite instead: the claim is that it is *at most* small.
        p_pair = wilcoxon(ra, re_).pvalue if len(re_) > 5 else np.nan
        rng = np.random.default_rng(0)
        boot = np.array([re_[rng.integers(0, len(re_), len(re_))].mean()
                         for _ in range(10000)])
        el_lo, el_hi = np.quantile(boot, [0.025, 0.975])

        # restriction of range: do the elite candidates actually differ in outcome?
        ratio_true = np.array([
            np.std(np.asarray(r['true_costs'])[: r['n_elite']])
            / (np.std(np.asarray(r['true_costs'])) + 1e-12) for r in recs])
        nd = np.array([r['n_distinguishable'] for r in recs])
        reg = np.array([r['regret_norm'] for r in recs])
        cap, s_ref, s_sig = attenuation_bound(data, recs)

        rows.append({
            'env': env, 'n': int(ok.sum()),
            'rho_all': ra.mean(), 'rho_all_se': ra.std(ddof=1) / np.sqrt(len(ra)),
            'rho_elite': re_.mean(), 'rho_elite_se': re_.std(ddof=1) / np.sqrt(len(re_)),
            'rho_elite_ci': [float(el_lo), float(el_hi)], 'p_all_vs_elite': p_pair,
            'n_dist': nd.mean(), 'frac_zero_dist': float((nd == 0).mean()),
            'true_spread_ratio': float(np.median(ratio_true)),
            'regret': float(reg.mean()),
            'rho_cap': cap, 's_ref': s_ref, 's_sig': s_sig,
        })
        resol = np.array([r['elite_range_over_err'] for r in recs])
        per_env[env] = (ra, re_, reg, nd, resol)

        # the same contrast measured against the env's own state space, which the
        # model's encoder plays no part in: if the elite ordering fails there too,
        # the failure cannot be an artifact of the latent metric
        sa = np.array([r.get('rho_all_state', np.nan) for r in recs], dtype=float)
        se = np.array([r.get('rho_elite_state', np.nan) for r in recs], dtype=float)
        rows[-1]['rho_all_state'] = float(np.nanmean(sa)) if not np.all(np.isnan(sa)) else np.nan
        rows[-1]['rho_elite_state'] = float(np.nanmean(se)) if not np.all(np.isnan(se)) else np.nan
        pw_elite[env] = pairwise_accuracy(recs, elite_only=True)
        pw_all[env] = pairwise_accuracy(recs, elite_only=False)

    if not rows:
        print('no data')
        return

    hdr = (f"{'env':9s} {'n':>3s} {'rho_all':>14s} {'rho_elite':>14s} "
           f"{'rho_elite 95% CI':>18s} {'p(all>el)':>10s} {'cap':>5s} "
           f"{'n_dist':>7s} {'0-dist':>7s} {'sd_true_el/all':>14s} {'regret':>7s}")
    print('\n' + hdr)
    print('-' * len(hdr))
    for r in rows:
        lo, hi = r['rho_elite_ci']
        print(f"{LABEL[r['env']]:9s} {r['n']:3d} "
              f"{r['rho_all']:+.3f}+-{r['rho_all_se']:.3f} "
              f"{r['rho_elite']:+.3f}+-{r['rho_elite_se']:.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18s} {r['p_all_vs_elite']:10.1e} "
              f"{r['rho_cap']:5.2f} {r['n_dist']:7.2f} {r['frac_zero_dist']:7.0%} "
              f"{r['true_spread_ratio']:14.2f} {r['regret']:7.2f}")

    if any(not np.isnan(r.get('rho_all_state', np.nan)) for r in rows):
        print('\nSame contrast with the encoder removed, using distance in the '
              "environment's own\nstate space as the reference:")
        print(f"{'env':9s} {'rho_all (state)':>16s} {'rho_elite (state)':>18s}")
        for r in rows:
            if np.isnan(r.get('rho_all_state', np.nan)):
                continue
            print(f"{LABEL[r['env']]:9s} {r['rho_all_state']:+16.3f} "
                  f"{r['rho_elite_state']:+18.3f}")

    print('\nReading: rho_all is the model ordering the whole candidate range; '
          'rho_elite the same\nmodel ordering only the elite set CEM refits on. '
          'The CI is a bootstrap interval on\nrho_elite -- the claim is that it '
          'is at most small, which a failed test against\nzero would not '
          "establish. 'cap' is the largest correlation reachable given noise in\n"
          "the true cost, so rho_elite far below cap is the model's limit, not "
          "the\nmeasurement's. 'n_dist' is how many of the 30 elite candidates "
          "the model separates\nfrom its own pick, and '0-dist' the share of "
          'episodes where it separates none.\nsd_true_el/all near 1 means the '
          'elite candidates really do lead to different\noutcomes, so failing to '
          'order them is a failure and not a harmless tie.')

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    envs = [r['env'] for r in rows]
    x = np.arange(len(envs))

    ax = axes[0, 0]
    w = 0.36
    ax.bar(x - w / 2, [r['rho_all'] for r in rows], w,
           yerr=[r['rho_all_se'] for r in rows], capsize=3,
           color='#4C72B0', label='all sampled candidates')
    ax.bar(x + w / 2, [r['rho_elite'] for r in rows], w,
           yerr=[r['rho_elite_se'] for r in rows], capsize=3,
           color='#C44E52', label='elite set only (what CEM refits on)')
    for i, r in enumerate(rows):
        if not np.isnan(r['rho_cap']):
            ax.hlines(r['rho_cap'], i - 0.45, i + 0.45, color='0.35',
                      ls=':', lw=1.4, zorder=5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[e] for e in envs])
    ax.set_ylabel('Spearman corr. of model cost vs true cost')
    ax.set_title('(a) The model orders candidates it does not choose between')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8, loc='upper left')
    ax.text(0.98, 0.965, 'dotted line: ceiling set by noise in the true cost',
            transform=ax.transAxes, fontsize=7.5, color='0.35',
            ha='right', va='top')

    ax = axes[0, 1]
    for env in envs:
        r = pw_elite.get(env)
        if r is None:
            continue
        c, a, _, _ = r
        ax.plot(c, 100 * a, 'o-', ms=4, label=LABEL[env])
    ax.axhspan(45, 55, color='0.85', zorder=0)
    ax.axhline(50, color='k', ls='--', lw=1.2, zorder=1)
    ax.set_xscale('log')
    # a fixed, generous range: the finding is that these curves stay pinned near
    # chance, which auto-scaling to their own 49-56% span would hide
    ax.set_ylim(40, 100)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("model's predicted cost gap  (elite-set std units)")
    ax.set_ylabel('elite pairs ordered correctly (%)')
    ax.set_title('(b) Accuracy vs how strongly the model prefers a candidate')
    ax.text(0.99, 0.10, 'grey band: within 5 points of chance', color='0.3',
            fontsize=7.5, transform=ax.transAxes, ha='right', va='bottom')
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[1, 0]
    parts = [per_env[e][4] for e in envs]
    ax.boxplot(parts, tick_labels=[LABEL[e] for e in envs], showfliers=False)
    ax.axhline(1.0, color='#C44E52', ls='--', lw=1.2)
    ax.set_yscale('log')
    ax.set_ylabel('elite cost range / model cost error')
    ax.set_title("(c) The elite set is narrower than the model's own error")
    ax.text(0.5, 0.04, 'the model could separate its elite candidates only above '
                       'the dashed line',
            transform=ax.transAxes, fontsize=7.5, color='0.3',
            ha='center', va='bottom')

    ax = axes[1, 1]
    parts = [per_env[e][2] for e in envs]
    ax.boxplot(parts, tick_labels=[LABEL[e] for e in envs], showfliers=False)
    ax.set_ylabel("regret of the model's pick (true-cost std)")
    ax.set_title('(d) The cost of picking the wrong member of the elite set')
    ax.axhline(0, color='k', lw=0.8)

    fig.suptitle('A latent world model cannot rank the candidates its planner '
                 'chooses between', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = FIG_DIR / 'rank_fidelity.png'
    fig.savefig(out, dpi=160)
    print(f'\nsaved -> {out}')

    with open('outputs/runs/direction1_rank_fidelity_summary.json', 'w') as f:
        json.dump(rows, f, indent=2, default=float)

    # pairwise accuracy inside the elite band, the single most quotable number
    print('\nPairwise accuracy restricted to the elite set (chance = 50%):')
    for env in envs:
        r = pw_elite.get(env)
        if r is None:
            continue
        c, a, n, gaps = r
        lo = a[: max(1, len(a) // 2)]
        print(f'  {LABEL[env]:9s} lower half of the gap range: {100 * lo.mean():.1f}%  '
              f'(top bin {100 * a[-1]:.1f}%, n_pairs={int(n.sum())})')


if __name__ == '__main__':
    main()
