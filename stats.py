"""Statistical machinery: Friedman + Iman-Davenport, Nemenyi CD, Holm, Wilcoxon, Cliff's delta."""
import numpy as np
from scipy import stats

# Nemenyi critical values q_alpha for alpha = 0.05, k = 2..20
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
       16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def friedman(M, higher_is_better=True):
    """M: (n_datasets, k_methods). Returns ranks, chi2, p, Iman-Davenport F and p."""
    M = np.asarray(M, float)
    n, k = M.shape
    R = np.array([stats.rankdata(-row if higher_is_better else row) for row in M])
    avg = R.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4)
    p_chi = 1 - stats.chi2.cdf(chi2, k - 1)
    denom = n * (k - 1) - chi2
    if denom <= 0:
        F, p_F = np.inf, 0.0
    else:
        F = (n - 1) * chi2 / denom
        p_F = 1 - stats.f.cdf(F, k - 1, (k - 1) * (n - 1))
    return avg, float(chi2), float(p_chi), float(F), float(p_F)


def nemenyi_cd(k, n, alpha=0.05):
    q = Q05.get(k, 3.544)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def holm(avg_ranks, control_idx, n, k):
    """Holm-corrected post-hoc against a control method (Demsar 2006)."""
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    out = []
    for i, r in enumerate(avg_ranks):
        if i == control_idx:
            continue
        z = (r - avg_ranks[control_idx]) / se
        out.append([i, float(z), float(2 * (1 - stats.norm.cdf(abs(z))))])
    out.sort(key=lambda a: a[2])
    m = len(out)
    for j, row in enumerate(out):
        row.append(min(1.0, row[2] * (m - j)))
    for j in range(1, m):
        out[j][3] = max(out[j][3], out[j - 1][3])
    return out       # [idx, z, p_raw, p_holm]


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    d = (gt - lt) / (len(a) * len(b))
    ad = abs(d)
    lab = ("Negligible" if ad < 0.147 else "Small" if ad < 0.33
           else "Medium" if ad < 0.474 else "Large")
    return float(d), lab


def wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        return 1.0
