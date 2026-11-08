"""One large-N trial on the OBP synthetic bandit; prints a JSON record.

Run by experiments/obp/run.py in a fresh process so that the peak resident
memory covers a single trial.
"""
import argparse
import json
import sys
import time

import numpy as np
from obp.dataset import SyntheticBanditDataset, logistic_reward_function

import oips

EPSILONS = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
N_ACTIONS = 10
N_FOLD = 5
REWARD_SEED = 12345


def peak_rss_mib():
    try:
        import resource
    except ImportError:  # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    # The reward coefficients are the same in every trial; contexts, actions
    # and rewards are drawn from the trial seed.
    def reward_function(context, action_context, random_state=None):
        return logistic_reward_function(context=context, action_context=action_context, random_state=REWARD_SEED)

    dataset = SyntheticBanditDataset(n_actions=N_ACTIONS, dim_context=5, reward_type="binary",
                                     reward_function=reward_function, random_state=args.seed)
    q = dataset.obtain_batch_bandit_feedback(n_rounds=args.n)["expected_reward"]
    eval_actions = np.argmax(q, axis=1)
    truth = float(np.mean(q[np.arange(args.n), eval_actions]))

    K, size = len(EPSILONS), args.n // len(EPSILONS)
    strata = [(k * size, (k + 1) * size) for k in range(K)]
    rng = np.random.RandomState(args.seed + 1_000_003)
    actions = eval_actions.copy()
    for (start, end), eps in zip(strata, EPSILONS):
        explore = rng.uniform(size=size) < eps
        actions[start:end][explore] = rng.randint(0, N_ACTIONS, size=explore.sum())
    r = dataset.sample_reward_given_expected_reward(expected_reward=q, action=actions).astype(float)
    pi_e = (actions == eval_actions).astype(float)
    pi = (1.0 - EPSILONS[:, None]) * pi_e + EPSILONS[:, None] / N_ACTIONS
    pi_own = np.concatenate([pi[k, start:end] for k, (start, end) in enumerate(strata)])
    n = np.full(K, size)

    def folds():
        return oips.make_folds(strata, N_FOLD, [0] * K)

    calls = {
        "ips": lambda: oips.ips(r, pi_e, pi_own),
        "bips": lambda: oips.balanced_ips(r, pi_e, pi, n),
        "wips": lambda: oips.weighted_ips(r, pi_e, pi_own, folds()),
        "oips": lambda: oips.optimal_ips(r, pi_e, pi, folds()),
    }
    estimates, seconds = {}, {}
    for name, call in calls.items():
        start = time.perf_counter()
        estimates[name] = call()
        seconds[name] = time.perf_counter() - start
    print(json.dumps({"truth": truth, "estimates": estimates, "seconds": seconds, "peak_rss_mib": peak_rss_mib()}))


if __name__ == "__main__":
    main()
