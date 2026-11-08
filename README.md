# Variational Approach to Optimal IPS Estimator for Multi-logger Off-Policy Evaluation

Code for the NeurIPS 2026 paper by Joon Suk Huh and Junghoon Seo.

- `oips.py`: the estimators
- `experiments/`: scripts for the experiments (Section 5, Appendices A-D and F)
- `results/`: the per-trial estimates behind the figures and tables in the paper

## TL;DR

- With data from several logging policies, neither balanced IPS (bIPS) nor
  weighted IPS (wIPS) is consistently better. oIPS instead uses the
  sample-dependent weights that minimize the variance subject to unbiasedness.
- The weights have a closed form, derived by calculus of variations. Instead of
  a reward model, oIPS needs only a K × K matrix and a K-vector of moments,
  where K is the number of loggers.
- The doubly robust version, DR-oIPS, has the same asymptotic variance as
  DR-bIPS when the reward model is consistent, and can stay efficient when the
  reward model is misspecified in certain ways.
- In the experiments, oIPS generally improves on IPS, bIPS and wIPS, and DR-oIPS
  often has lower error than DR-bIPS.

## Setup

Python 3.13 with the pinned packages:

```bash
pip install -r requirements.txt
```

or Docker:

```bash
docker build -t optimal-ips .
docker run --rm -it -v "$PWD:/app" optimal-ips
```

Run all commands from the repository root.

## Estimators

| Function | Estimator | Paper |
| --- | --- | --- |
| `ips(r, pi_e, pi_own)` | IPS | Eq. (2.1) |
| `balanced_ips(r, pi_e, pi, n)` | balanced IPS | Eq. (2.2) |
| `weighted_ips(r, pi_e, pi_own, folds)` | weighted IPS | Eq. (2.3) |
| `optimal_ips(r, pi_e, pi, folds)` | oIPS | Eq. (3.6) |
| `dr_estimates(r, pi_e, folds, nuisances)` | DR-bIPS and DR-oIPS | Eqs. (2.4) and (B.2) |

The samples are grouped by logger: logger k's samples are `start:end`, where
`strata[k] = (start, end)`. For sample j,

- `r[j]` is the reward,
- `pi_e[j]` is the probability that the evaluation policy takes the logged action,
- `pi[k, j]` is the probability that logger k takes it,
- `pi_own[j]` is that probability for the logger that collected sample j.

`n[k]` is the number of samples from logger k, and `folds` are the
cross-fitting folds returned by `make_folds`.

```python
import numpy as np
import oips

rng = np.random.default_rng(0)
n_actions = 5
q = rng.random(n_actions)                      # mean reward of each action
pi_e = np.eye(n_actions)[0]                    # evaluation policy: always action 0
loggers = [0.9 * pi_e + 0.1 / n_actions,       # logger 0: close to pi_e
           np.full(n_actions, 1 / n_actions)]  # logger 1: uniform
strata = [(0, 2000), (2000, 10000)]            # samples of each logger

a = np.concatenate([rng.choice(n_actions, size=end - start, p=p)
                    for (start, end), p in zip(strata, loggers)])
r = rng.binomial(1, q[a]).astype(float)
pi = np.stack([p[a] for p in loggers])

folds = oips.make_folds(strata, n_splits=5, seeds=[0, 1])
print(oips.optimal_ips(r, pi_e[a], pi, folds))  # 0.6355
print(q[0])                                     # true value: 0.6370
```

## Reproducing the paper

| Paper | Scripts | Reference output |
| --- | --- | --- |
| Figures 1-11, Table 2 | `experiments/uci/run.py`, `summarize.py`, `plot.py` | `results/uci` |
| Table 3 | `experiments/uci/runtime.py` | |
| Tables 4 and 5 | `experiments/obp/run.py` | `results/obp` |

The figures and tables can be rebuilt from the reference output in under a
minute:

```bash
python -m experiments.uci.summarize results/uci  # prints Table 2 and the counts in Appendices A-C
python -m experiments.uci.plot results/uci       # writes figures/fig1.pdf ... fig11.pdf
python -m experiments.obp.run --trials-csv results/obp/trials.csv  # prints Tables 4 and 5
```

### UCI benchmarks (Figures 1-11, Table 2)

`data/uci` contains the four UCI datasets and `data/trials` the 200 trials used
in the paper. Each trial stores a random 70% OPE pool and the actions of the
deterministic policy fitted on the other 30%. `experiments/uci/prepare.py`
rebuilds these files.

The settings are in `experiments/uci/configs`:

| File | Setting | Paper |
| --- | --- | --- |
| `main.jsonl` | K = 2, η = 0.95, seven ratios n1/n2 | Figures 1-3, Table 2 |
| `multi_logger.jsonl` | K = 3 and K = 5 | Figures 4-7 |
| `mixtures.jsonl` | K = 2, η = 0.80 and 0.65 | Figures 8-11 |

```bash
python -m experiments.uci.run --output output/uci
python -m experiments.uci.summarize output/uci
python -m experiments.uci.plot output/uci --out figures
python -m experiments.uci.compare output/uci  # differences from results/uci
```

The full run (184 settings with 200 trials each) takes about an hour with 13
processes on an 8-core desktop. Each setting is saved when it finishes, and a
restarted run skips the finished ones. `--configs`, `--datasets` and `--trials`
select a subset:

```bash
python -m experiments.uci.run --configs experiments/uci/configs/main.jsonl \
    --datasets optdigits --trials 20 --output output/quick
```

### Runtime (Table 3)

```bash
python -m experiments.uci.runtime
```

prints the Relative-RMSE and the time per estimator call with K = 10 and
K = 100 loggers on Letter. Times depend on the machine; the paper's were
measured on an Apple M1.

### Large-N synthetic experiment (Tables 4 and 5)

This experiment uses the synthetic bandit of Open Bandit Pipeline (OBP) 0.5.7,
which requires Python 3.10. OBP lists torch as a dependency, but the synthetic
bandit does not use it, so OBP is installed without dependencies and patched to
import without torch:

```bash
python3.10 -m venv .venv-obp
source .venv-obp/bin/activate
pip install -r experiments/obp/requirements.txt
pip install --no-deps obp==0.5.7
python experiments/obp/patch_obp.py
python -m experiments.obp.run  # writes output/obp, prints Tables 4 and 5
```

or with Docker:

```bash
docker build -f experiments/obp/Dockerfile -t optimal-ips-obp .
docker run --rm -v "$PWD:/app" optimal-ips-obp
```

Each trial runs in its own process, so the peak memory in Table 5 is that of a
single trial.

## Repository layout

```
oips.py                 estimators
experiments/uci/
    run.py              trials for every setting
    summarize.py        Relative-RMSE, bootstrap intervals, Table 2
    plot.py             Figures 1-11
    compare.py          rerun vs. reference
    runtime.py          Table 3
    prepare.py          data/trials from the raw UCI files
    common.py           data loading, configuration ids, seeds
    configs/            settings
experiments/obp/        Tables 4 and 5
data/uci/               UCI datasets
data/trials/            trials used in the paper
results/                reference results
```

## Citation

```bibtex
@inproceedings{huh2026variational,
  title     = {Variational Approach to Optimal {IPS} Estimator for Multi-logger Off-Policy Evaluation},
  author    = {Huh, Joon Suk and Seo, Junghoon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is under the MIT license. The UCI datasets and the files derived from
them are under CC BY 4.0; see `LICENSE`.
