"""Relative-RMSE (Eq. 5.1) with 95% paired percentile-bootstrap intervals.

Reads <results>/trials/*.csv and writes <results>/summary.csv and
<results>/paired_differences.csv, then prints the comparisons quoted in
Appendices A-C and Table 2.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.uci.common import DATASETS, METHODS, stable_seed

POINT = ["setting", "config_id", "dataset", "estimate_pi_b", "num_loggers", "ratio_label"]
COMPARISONS = [("oips", "ips"), ("oips", "bips"), ("oips", "wips"), ("dr_oips", "dr_bips")]


def rel_rmse(est, truth):
    return np.sqrt(np.mean((est - truth) ** 2, axis=-1)) / np.mean(truth, axis=-1)


def summarize_point(trials, n_boot, seed):
    trials = trials.sort_values("trial")
    truth = trials["truth"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(trials), size=(n_boot, len(trials)))
    rows, boot = {}, {}
    for m in METHODS:
        est = trials[m].to_numpy()
        boot[m] = rel_rmse(est[idx], truth[idx])
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        bias = np.mean(est - truth)
        rows[m] = {"method": m, "rel_rmse": rel_rmse(est, truth), "ci_low": lo, "ci_high": hi,
                   "bias_sq": bias * bias, "sd": np.std(est, ddof=1)}
    pairs = []
    for m, base in COMPARISONS:
        lo, hi = np.percentile(boot[m] - boot[base], [2.5, 97.5])
        pairs.append({"method": m, "baseline": base, "diff": rows[m]["rel_rmse"] - rows[base]["rel_rmse"],
                      "ci_low": lo, "ci_high": hi})
    return list(rows.values()), pairs


def report(summary, pairs):
    for estimated, label in [(False, "known"), (True, "estimated")]:
        p = pairs[pairs.estimate_pi_b == estimated]
        non_dr, dr = p[p.method == "oips"], p[p.method == "dr_oips"]
        print(f"{label} propensities: oIPS beats IPS/bIPS/wIPS in {(non_dr['diff'] < 0).sum()}/{len(non_dr)}; "
              f"DR-oIPS beats DR-bIPS in {(dr['diff'] < 0).sum()}/{len(dr)} "
              f"(95% interval below zero in {(dr.ci_high < 0).sum()})")
    dr = pairs[pairs.method == "dr_oips"]
    fig3 = dr[(dr.setting == "main_eta_0.95") & dr.estimate_pi_b]
    print(f"Figure 3 setting: DR-oIPS beats DR-bIPS in {(fig3['diff'] < 0).sum()}/{len(fig3)}; "
          f"all settings: {(dr['diff'] < 0).sum()}/{len(dr)} (interval below zero in {(dr.ci_high < 0).sum()})")
    for k in (5, 3):
        sub = dr[(dr.num_loggers == k) & dr.estimate_pi_b]
        print(f"K={k}, estimated: DR-oIPS beats DR-bIPS on {(sub['diff'] < 0).sum()}/{len(sub)} datasets")

    t2 = summary[(summary.setting == "main_eta_0.95") & ~summary.estimate_pi_b
                 & summary.ratio_label.isin(["0.1", "1", "10"])]
    print("\nTable 2: squared bias / empirical SD (known propensities, eta = 0.95)")
    print(f"{'n1/n2':>6} {'dataset':>10} " + " ".join(f"{m:>10}" for m in METHODS))
    for ratio in ["0.1", "1", "10"]:
        for d in DATASETS:
            cell = t2[(t2.ratio_label == ratio) & (t2.dataset == d)].set_index("method").loc[METHODS]
            print(f"{ratio:>6} {d:>10} " + " ".join(f"{x:10.3e}" for x in cell.bias_sq))
            print(f"{'':>17} " + " ".join(f"{x:10.6f}" for x in cell.sd))
    print(f"max squared bias / variance: {(t2.bias_sq / t2.sd ** 2).max():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/uci"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    trials = pd.concat([pd.read_csv(f, dtype={"ratio_label": str}) for f in sorted((args.results / "trials").glob("*.csv"))])
    summary, pairs = [], []
    for key, group in trials.groupby(POINT, sort=True):
        meta = dict(zip(POINT, key))
        rows, diffs = summarize_point(group, args.bootstrap, stable_seed(args.seed, meta["config_id"], meta["dataset"]))
        summary += [{**meta, **row} for row in rows]
        pairs += [{**meta, **row} for row in diffs]
    summary, pairs = pd.DataFrame(summary), pd.DataFrame(pairs)
    summary.to_csv(args.results / "summary.csv", index=False, lineterminator="\n")
    pairs.to_csv(args.results / "paired_differences.csv", index=False, lineterminator="\n")
    print(f"{trials.groupby(POINT).ngroups} points, {len(trials)} trials\n")
    report(summary, pairs)


if __name__ == "__main__":
    main()
