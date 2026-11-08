import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA = Path("data")
DATASETS = ["letter", "optdigits", "pendigits", "sat"]
METHODS = ["ips", "bips", "wips", "oips", "dr_bips", "dr_oips"]


def load_uci(name):
    """Features and integer labels of a UCI dataset (train and test files concatenated)."""
    root = DATA / "uci" / name
    if name == "letter":
        raw = np.genfromtxt(root / "letter-recognition.data", delimiter=",", dtype=str)
        return raw[:, 1:].astype(float), LabelEncoder().fit_transform(raw[:, 0])
    if name == "sat":
        data = np.r_[np.loadtxt(root / "sat.trn"), np.loadtxt(root / "sat.tst")]
        y = data[:, -1].astype(int)
        return data[:, :-1], np.where(y == 7, 5, y - 1)  # labels 1-5, 7 -> 0-5
    data = np.r_[np.loadtxt(root / f"{name}.tra", delimiter=","),
                 np.loadtxt(root / f"{name}.tes", delimiter=",")]
    return data[:, :-1], data[:, -1].astype(int)


def load_trials(name):
    """Per-trial OPE pool (row_indices), labels (y_gt) and pi_det actions (y_eval_pol)."""
    return pd.read_parquet(DATA / "trials" / f"{name}.parquet")


def stable_seed(*parts):
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little")


def config_id(cfg):
    """Name of a configuration and the hash that seeds its trials."""
    spec = {
        "setting": cfg["setting"],
        "ratio_label": cfg["ratio_label"],
        "estimate_pi_b": bool(cfg["estimate_pi_b"]),
        "num_loggers": len(cfg["exploration_probs"]),
        "n_fold": int(cfg["n_fold"]),
        "exploration_probs": [float(x) for x in cfg["exploration_probs"]],
        "stratum_ratio": [float(x) for x in cfg["stratum_ratio"]],
    }
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    policy = "estimated" if spec["estimate_pi_b"] else "known"
    ratio = "-".join(f"{x:.4f}".rstrip("0").rstrip(".").replace(".", "p") for x in spec["stratum_ratio"])
    return f"{spec['setting']}__{policy}__k{spec['num_loggers']}__ratio_{ratio}__{digest[:10]}", digest


def read_configs(paths):
    return [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()]
