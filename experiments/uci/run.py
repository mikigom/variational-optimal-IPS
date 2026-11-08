"""Run the UCI multi-logger experiments (Section 5, Appendices A-D).

Every (configuration, dataset) pair is one point. Its trials are written to
<output>/trials/<config_id>__<dataset>.csv, and points that already have a
complete file are skipped, so an interrupted run can be resumed.
"""
import os

# One worker process per point; keep BLAS single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import oips
from experiments.uci.common import DATASETS, config_id, load_trials, load_uci, read_configs, stable_seed

# Logistic regressions use the scikit-learn defaults (lbfgs, 100 iterations),
# which often stop before convergence on these unscaled features.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_strata(n, ratios):
    bounds = [0]
    for ratio in ratios[:-1]:
        bounds.append(bounds[-1] + int(n * ratio))
    bounds.append(n)
    strata = list(zip(bounds[:-1], bounds[1:]))
    return np.array([end - start for start, end in strata]), strata


def log_actions(eval_actions, strata, eps, n_actions, seed):
    """Logger k replaces pi_det's action by a uniform one with probability eps[k]."""
    rng = np.random.default_rng(seed)
    actions = eval_actions.copy()
    for (start, end), p in zip(strata, eps):
        explore = rng.random(end - start) <= p
        actions[start:end][explore] = rng.integers(0, n_actions, size=explore.sum())
    return actions


def known_propensities(actions, eval_actions, eps, n_actions):
    match = (actions == eval_actions).astype(float)
    return np.vstack([(1.0 - p) * match + p / n_actions for p in eps])


def fit_policy(states, actions):
    if np.unique(actions).size < 2:
        return None
    return LogisticRegression(solver="lbfgs", max_iter=100).fit(states, actions)


def policy_proba(model, states, actions, n_actions):
    """Estimated propensities of the logged actions: 1% uniform smoothing, floor 1e-3."""
    if model is None:
        p = np.full(len(actions), 1.0 / n_actions)
    else:
        full = np.zeros((len(actions), n_actions))
        full[:, model.classes_] = model.predict_proba(states)
        full = 0.99 * full + 0.01 / n_actions
        p = full[np.arange(len(actions)), actions]
    return np.clip(p, 1e-3, 1.0)


def estimated_propensities(states, actions, strata, n_actions, seed):
    """Appendix A: own-logger propensities are 2-fold out-of-fold; row k of ``pi``
    comes from a model fitted on logger k's whole stratum."""
    pi_own = np.empty(len(actions))
    pi = np.empty((len(strata), len(actions)))
    for k, (start, end) in enumerate(strata):
        idx = np.arange(start, end)
        _, counts = np.unique(actions[idx], return_counts=True)
        if counts.size >= 2 and counts.min() >= 2:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=stable_seed(seed, "behavior-own-oof", k))
            for train, test in skf.split(states[idx], actions[idx]):
                model = fit_policy(states[idx[train]], actions[idx[train]])
                pi_own[idx[test]] = policy_proba(model, states[idx[test]], actions[idx[test]], n_actions)
        else:
            pi_own[idx] = 1.0 / n_actions
        pi[k] = policy_proba(fit_policy(states[idx], actions[idx]), states, actions, n_actions)
    return pi_own, pi


def features(states, actions, n_actions):
    one_hot = np.zeros((len(actions), n_actions))
    one_hot[np.arange(len(actions)), actions] = 1.0
    return np.hstack([states, one_hot])


def fit_reward(states, actions, r, n_actions):
    if np.unique(r).size < 2:
        return float(np.mean(r))
    return LogisticRegression(solver="lbfgs", max_iter=100).fit(features(states, actions, n_actions), r)


def predict_reward(model, states, actions, n_actions):
    if isinstance(model, float):
        return np.full(len(actions), model)
    return model.predict_proba(features(states, actions, n_actions))[:, 1]


def dr_nuisance(states, actions, eval_actions, r, train, test, n_actions, eps, estimated):
    """Fit q_hat (and pi_hat if needed) on one training fold, Appendix B.2."""
    tr, te = np.concatenate(train), np.concatenate(test)
    if estimated:
        pi_tr, pi_te = np.empty((len(train), tr.size)), np.empty((len(train), te.size))
        for k, idx in enumerate(train):
            model = fit_policy(states[idx], actions[idx])
            pi_tr[k] = policy_proba(model, states[tr], actions[tr], n_actions)
            pi_te[k] = policy_proba(model, states[te], actions[te], n_actions)
    else:
        pi_tr = known_propensities(actions[tr], eval_actions[tr], eps, n_actions)
        pi_te = known_propensities(actions[te], eval_actions[te], eps, n_actions)
    model = fit_reward(states[tr], actions[tr], r[tr], n_actions)
    return (pi_tr, pi_te,
            predict_reward(model, states[tr], actions[tr], n_actions),
            predict_reward(model, states[te], actions[te], n_actions),
            predict_reward(model, states[te], eval_actions[te], n_actions))


def run_trial(cfg, digest, dataset, trial, X, row, n_actions, seed):
    labels = np.asarray(row["y_gt"], dtype=int)
    eval_actions = np.asarray(row["y_eval_pol"], dtype=int)
    states = X[np.asarray(row["row_indices"], dtype=int)]
    eps = cfg["exploration_probs"]
    sizes, strata = make_strata(len(labels), cfg["stratum_ratio"])

    # Seeds are hashed from fixed labels; changing a label changes the random stream.
    fold_seed = stable_seed(seed, digest, dataset, trial, "folds")
    actions = log_actions(eval_actions, strata, eps, n_actions, stable_seed(seed, digest, dataset, trial, "logging"))
    pi_e = (actions == eval_actions).astype(float)
    r = (actions == labels).astype(float)

    def folds(n_splits, folds_seed):
        return oips.make_folds(strata, n_splits, [stable_seed(folds_seed, "logger-fold", k) for k in range(len(strata))])

    cv_folds = folds(cfg["n_fold"], stable_seed(fold_seed, "non-dr-five-fold"))
    if cfg["estimate_pi_b"]:
        pi_own, pi = estimated_propensities(states, actions, strata, n_actions, fold_seed)
    else:
        pi = known_propensities(actions, eval_actions, eps, n_actions)
        pi_own = np.concatenate([pi[k, start:end] for k, (start, end) in enumerate(strata)])

    est = {
        "ips": oips.ips(r, pi_e, pi_own),
        "bips": oips.balanced_ips(r, pi_e, pi, sizes),
        "wips": oips.weighted_ips(r, pi_e, pi_own, cv_folds),
        "oips": oips.optimal_ips(r, pi_e, pi, cv_folds),
    }
    dr_folds = folds(2, fold_seed)
    nuisances = [dr_nuisance(states, actions, eval_actions, r, train, test, n_actions, eps, cfg["estimate_pi_b"])
                 for train, test in dr_folds]
    est["dr_bips"], est["dr_oips"] = oips.dr_estimates(r, pi_e, dr_folds, nuisances)
    return {"trial": trial, "truth": float(np.mean(labels == eval_actions)), **est}


def run_point(cfg, dataset, n_trials, seed, path):
    if path.exists() and len(pd.read_csv(path)) == n_trials:
        return "skipped"
    name, digest = config_id(cfg)
    X, y = load_uci(dataset)
    trials = load_trials(dataset)
    n_actions = int(y.max()) + 1
    rows = [run_trial(cfg, digest, dataset, t, X, trials.iloc[t], n_actions, seed) for t in range(n_trials)]
    meta = {"setting": cfg["setting"], "config_id": name, "dataset": dataset, "estimate_pi_b": cfg["estimate_pi_b"],
            "num_loggers": len(cfg["exploration_probs"]), "ratio_label": cfg["ratio_label"]}
    pd.DataFrame([{**meta, **row} for row in rows]).to_csv(path, index=False, lineterminator="\n")
    return "done"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=sorted(map(str, Path("experiments/uci/configs").glob("*.jsonl"))))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--output", type=Path, default=Path("output/uci"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    (args.output / "trials").mkdir(parents=True, exist_ok=True)
    points = [(cfg, d) for cfg in read_configs(args.configs) for d in args.datasets]
    print(f"{len(points)} points x {args.trials} trials, {args.workers} workers")
    start = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {}
        for cfg, dataset in points:
            path = args.output / "trials" / f"{config_id(cfg)[0]}__{dataset}.csv"
            jobs[pool.submit(run_point, cfg, dataset, args.trials, args.seed, path)] = path.stem
        for i, job in enumerate(as_completed(jobs), 1):
            print(f"[{i}/{len(jobs)}] {job.result()} {jobs[job]} ({(time.time() - start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
