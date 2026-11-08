"""Relative-RMSE and estimator runtime with K = 10 and K = 100 loggers on Letter
(Appendix F.2, Table 3).

Loggers mix pi_det with the uniform policy, with exploration probabilities
evenly spaced on [0.05, 0.95], and share the OPE pool equally. Logging
policies are known; wIPS and oIPS use Z = 5 cross-fitting folds. Timings
include building the folds.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import oips
from experiments.uci.common import load_trials

ESTIMATORS = ["ips", "bips", "wips", "oips"]


def run(trials, n_loggers, n_fold, n_actions):
    eps = np.round(np.linspace(0.05, 0.95, n_loggers), 5)
    rows = []
    for t, row in trials.iterrows():
        labels, eval_actions = row["y_gt"], row["y_eval_pol"]
        size = len(labels) // n_loggers
        strata = [(k * size, (k + 1) * size if k < n_loggers - 1 else len(labels)) for k in range(n_loggers)]
        n = np.array([end - start for start, end in strata])

        rng = np.random.RandomState(42)
        actions = eval_actions.copy()
        for (start, end), p in zip(strata, eps):
            explore = rng.uniform(size=end - start) <= p
            actions[start:end][explore] = rng.choice(n_actions, size=explore.sum())
        pi_e = (actions == eval_actions).astype(float)
        r = (actions == labels).astype(float)
        pi = (1.0 - eps[:, None]) * pi_e + eps[:, None] / n_actions
        pi_own = np.concatenate([pi[k, start:end] for k, (start, end) in enumerate(strata)])

        def folds():
            return oips.make_folds(strata, n_fold, [0] * n_loggers)

        calls = {
            "ips": lambda: oips.ips(r, pi_e, pi_own),
            "bips": lambda: oips.balanced_ips(r, pi_e, pi, n),
            "wips": lambda: oips.weighted_ips(r, pi_e, pi_own, folds()),
            "oips": lambda: oips.optimal_ips(r, pi_e, pi, folds()),
        }
        out = {"loggers": n_loggers, "trial": t, "truth": np.mean(labels == eval_actions)}
        for name, call in calls.items():
            start = time.perf_counter()
            out[name] = call()
            out[f"{name}_ms"] = 1e3 * (time.perf_counter() - start)
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loggers", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--n-fold", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("output/runtime.csv"))
    args = parser.parse_args()

    trials = load_trials("letter")
    df = pd.concat([run(trials, k, args.n_fold, 26) for k in args.loggers])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, lineterminator="\n")

    print(f"{'':>6} " + " ".join(f"{'K=' + str(k) + ' Rel-RMSE':>16} {'time (ms)':>14}" for k in args.loggers))
    for name in ESTIMATORS:
        cells = []
        for k in args.loggers:
            d = df[df.loggers == k]
            rel = np.sqrt(np.mean((d[name] - d.truth) ** 2)) / d.truth.mean()
            ms = d[f"{name}_ms"]
            cells.append(f"{rel:16.4f} {ms.mean():8.2f} +- {1.96 * ms.std() / np.sqrt(len(ms)):.2f}")
        print(f"{name:>6} " + " ".join(cells))


if __name__ == "__main__":
    main()
