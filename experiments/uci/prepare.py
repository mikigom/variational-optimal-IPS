"""Build data/trials/<dataset>.parquet from the raw UCI files (Section 5.1).

For each trial the data are split at random: a multinomial logistic regression
fitted on 30% defines the deterministic policy pi_det, and the remaining 70% is
the OPE pool. Each row stores the pool indices, their labels and pi_det's actions.

The files used for the paper are included in the repository.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments.uci.common import DATA, DATASETS, load_uci

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--ope-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name in args.datasets:
        X, y = load_uci(name)
        rows = []
        for t in range(args.trials):
            pool, rest = train_test_split(np.arange(len(y)), test_size=1 - args.ope_ratio,
                                          random_state=args.seed + t, shuffle=True)
            clf = LogisticRegression(solver="lbfgs", max_iter=100).fit(X[rest], y[rest])
            rows.append({"y_gt": y[pool].astype(np.int32), "y_eval_pol": clf.predict(X[pool]).astype(np.int32),
                         "row_indices": pool.astype(np.int32)})
        (DATA / "trials").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(DATA / "trials" / f"{name}.parquet")
        print(f"{name}: {args.trials} trials, pool size {len(rows[0]['y_gt'])}")


if __name__ == "__main__":
    main()
