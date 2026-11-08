"""Compare a rerun with the reference results.

    python -m experiments.uci.compare output/uci [results/uci]

Run experiments.uci.summarize on the rerun first.
"""
import sys
from pathlib import Path

import pandas as pd

from experiments.uci.common import METHODS


def main():
    new, ref = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "results/uci")
    files = sorted((new / "trials").glob("*.csv"))
    diffs = []
    for f in files:
        a = pd.read_csv(f, float_precision="round_trip").set_index("trial")
        b = pd.read_csv(ref / "trials" / f.name, float_precision="round_trip").set_index("trial").loc[a.index]
        diffs.append((a[METHODS] - b[METHODS]).abs().assign(estimate_pi_b=a.estimate_pi_b, num_loggers=a.num_loggers))
    d = pd.concat(diffs)
    print(f"{len(files)} points, {len(d)} trials")
    print("max |estimate - reference| per trial:")
    print(d.groupby(["estimate_pi_b", "num_loggers"])[METHODS].max().to_string(float_format="%.1e"))

    key = ["config_id", "dataset", "method"]
    s = pd.read_csv(new / "summary.csv").merge(pd.read_csv(ref / "summary.csv"), on=key, suffixes=("", "_ref"))
    s["rel_diff"] = (s.rel_rmse - s.rel_rmse_ref).abs() / s.rel_rmse_ref
    print("\nRelative-RMSE, |new - reference| / reference:")
    print(s.groupby(["estimate_pi_b", "method"]).rel_diff.agg(["median", "max"]).unstack(0).to_string(float_format="%.2e"))
    inside = ((s.rel_rmse >= s.ci_low_ref) & (s.rel_rmse <= s.ci_high_ref)).mean()
    print(f"\nrerun Relative-RMSE inside the reference 95% interval: {inside:.1%} of {len(s)}")


if __name__ == "__main__":
    main()
