"""Large-N synthetic experiment with the Open Bandit Pipeline (Appendix F.3, Tables 4 and 5).

Each (N, seed) trial runs in its own process (experiments/obp/trial.py).
Writes <output>/trials.csv and <output>/summary.csv.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["ips", "bips", "wips", "oips"]


def run_trial(n, seed):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, "-m", "experiments.obp.trial", "--n", str(n), "--seed", str(seed)],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    record = json.loads(out)
    return [{"n": n, "seed": seed, "method": m, "estimate": record["estimates"][m], "truth": record["truth"],
             "seconds": record["seconds"][m], "peak_rss_mib": record["peak_rss_mib"]} for m in METHODS]


def summarize(trials, n_boot, seed):
    """Relative-RMSE with paired bootstrap intervals over seeds."""
    rng = np.random.RandomState(seed)
    rows = []
    for n, group in trials.groupby("n", sort=True):
        est = group.pivot(index="seed", columns="method", values="estimate")[METHODS].to_numpy()
        truth = group.groupby("seed").truth.first().to_numpy()
        boot = np.empty((n_boot, len(METHODS)))
        for b in range(n_boot):
            idx = rng.randint(0, len(truth), size=len(truth))
            boot[b] = np.sqrt(np.mean((est[idx] - truth[idx, None]) ** 2, axis=0)) / np.mean(truth[idx])
        rel = np.sqrt(np.mean((est - truth[:, None]) ** 2, axis=0)) / np.mean(truth)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for j, m in enumerate(METHODS):
            g = group[group.method == m]
            rows.append({"n": n, "method": m, "rel_rmse": rel[j], "ci_low": lo[j], "ci_high": hi[j],
                         "mean_ms": 1e3 * g.seconds.mean(), "max_peak_rss_mib": g.peak_rss_mib.max()})
    return pd.DataFrame(rows)


def report(summary):
    ns = sorted(summary.n.unique())
    print("Table 4: Relative-RMSE x 1e3 (95% CI)")
    print(f"{'':>5} " + "".join(f"{'N=' + str(n):>24}" for n in ns))
    for m in METHODS:
        cells = [f"{1e3 * r.rel_rmse:#.3g} ({1e3 * r.ci_low:#.3g}, {1e3 * r.ci_high:#.3g})"
                 for r in summary[summary.method == m].sort_values("n").itertuples()]
        print(f"{m:>5} " + "".join(f"{c:>24}" for c in cells))
    print()
    print("Table 5: mean runtime (ms) and largest peak memory (MiB)")
    print(f"{'N':>8} " + "".join(f"{m:>10}" for m in METHODS) + f"{'memory':>10}")
    for n, s in summary.groupby("n"):
        s = s.set_index("method").loc[METHODS]
        print(f"{n:>8} " + "".join(f"{t:10.3f}" for t in s.mean_ms) + f"{s.max_peak_rss_mib.max():10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("output/obp"))
    parser.add_argument("--trials-csv", type=Path, help="summarize an existing trials.csv instead of running")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.trials_csv:
        trials = pd.read_csv(args.trials_csv, float_precision="round_trip")
    else:
        rows = []
        for n in args.n:
            for seed in range(args.seeds):
                print(f"N={n} seed={seed}", flush=True)
                rows += run_trial(n, seed)
        trials = pd.DataFrame(rows)
        trials.to_csv(args.output / "trials.csv", index=False, lineterminator="\n")
    summary = summarize(trials, args.bootstrap, seed=20260724)
    summary.to_csv(args.output / "summary.csv", index=False, lineterminator="\n")
    report(summary)


if __name__ == "__main__":
    main()
