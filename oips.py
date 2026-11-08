"""Multi-logger IPS estimators.

Notation follows the paper. With N logged samples from K loggers:

    r       rewards, shape (N,)
    pi_e    pi_e(A_j | S_j), shape (N,)
    pi_own  propensity of each sample under the logger that collected it, shape (N,)
    pi      pi[k, j] = pi_k(A_j | S_j) for every logger k, shape (K, N)
    n       stratum sizes, shape (K,)

Cross-fitted estimators take ``folds``, a list of ``(train, test)`` pairs in
which ``train[k]`` and ``test[k]`` index logger k's samples (see ``make_folds``).
"""
import numpy as np
from sklearn.model_selection import KFold


def make_folds(strata, n_splits, seeds):
    """Split every stratum ``(start, end)`` into ``n_splits`` folds."""
    folds = [([], []) for _ in range(n_splits)]
    for (start, end), seed in zip(strata, seeds):
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for z, (train, test) in enumerate(kfold.split(np.arange(end - start))):
            folds[z][0].append(train + start)
            folds[z][1].append(test + start)
    return folds


def _sizes(index_lists):
    return np.array([idx.size for idx in index_lists], dtype=float)


def ips(r, pi_e, pi_own):
    """Naive IPS, Eq. (2.1)."""
    return float(np.mean(r * pi_e / pi_own))


def balanced_ips(r, pi_e, pi, n):
    """Balanced IPS, Eq. (2.2)."""
    pi_avg = (n / n.sum()) @ pi
    return float(np.mean(r * pi_e / pi_avg))


def weighted_ips(r, pi_e, pi_own, folds):
    """Weighted IPS, Eq. (2.3); the variances are estimated on the training folds."""
    score = r * pi_e / pi_own
    value = 0.0
    for train, test in folds:
        var = np.array([np.var(score[idx], ddof=1) for idx in train])
        var = np.where(var > 1e-12, var, 1.0)
        m = _sizes(test)
        gamma = (1.0 / var) / np.sum(m / var)
        sums = np.array([np.sum(score[idx]) for idx in test])
        value += (m.sum() / r.size) * np.dot(sums, gamma)
    return float(value)


def solve_alpha(signal, pi_e, pi, n):
    """Estimate T and c, Eq. (3.5), and return pinv(T) c.

    The training samples are grouped by logger: the first n[0] come from
    logger 0, the next n[1] from logger 1, and so on.
    """
    pi_avg = (n / n.sum()) @ pi
    T = np.empty((len(n), len(n)))
    c = np.empty(len(n))
    start = 0
    for k, size in enumerate(n.astype(int)):
        s = slice(start, start + size)
        T[k] = np.mean(pi[:, s] / pi_avg[s], axis=1)
        c[k] = np.mean(signal[s] * pi_e[s] / pi_avg[s])
        start += size
    return np.linalg.pinv(T, rcond=1e-15) @ c


def optimal_ips(r, pi_e, pi, folds):
    """Cross-fitted oIPS, Eq. (3.6), evaluated in the control-variate form (3.4)."""
    value = 0.0
    for train, test in folds:
        tr, te = np.concatenate(train), np.concatenate(test)
        alpha = solve_alpha(r[tr], pi_e[tr], pi[:, tr], _sizes(train))
        n_te = _sizes(test)
        pi_te = pi[:, te]
        pi_avg = (n_te / n_te.sum()) @ pi_te
        fold = np.sum(alpha) + np.mean((r[te] * pi_e[te] - alpha @ pi_te) / pi_avg)
        value += (te.size / r.size) * fold
    return float(value)


def dr_estimates(r, pi_e, folds, nuisances):
    """Cross-fitted DR-bIPS, Eq. (2.4), and DR-oIPS, Eq. (B.2).

    ``nuisances[z] = (pi_train, pi_test, q_train, q_test, q_e_test)`` holds the
    predictions of models fitted on the training part of fold z, with samples
    ordered as ``np.concatenate(train)`` and ``np.concatenate(test)``. ``q`` is
    q_hat(S, A) and ``q_e`` is q_hat(S, pi_e).
    """
    dr_bips = dr_oips = 0.0
    for (train, test), (pi_tr, pi_te, q_tr, q_te, q_e) in zip(folds, nuisances):
        tr, te = np.concatenate(train), np.concatenate(test)
        alpha = solve_alpha(r[tr] - q_tr, pi_e[tr], pi_tr, _sizes(train))
        n_te = _sizes(test)
        pi_avg = (n_te / n_te.sum()) @ pi_te
        resid = r[te] - q_te
        fold_bips = np.mean(resid * pi_e[te] / pi_avg + q_e)
        fold_oips = np.mean(q_e) + np.sum(alpha) + np.mean((resid * pi_e[te] - alpha @ pi_te) / pi_avg)
        dr_bips += (te.size / r.size) * fold_bips
        dr_oips += (te.size / r.size) * fold_oips
    return float(dr_bips), float(dr_oips)
