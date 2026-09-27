"""DF-GWO experiment driver: fold protocol, grouping stage, fitness probe, method registry."""
import zlib

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.decomposition import PCA

from common import load_dataset, fold_preprocess, smote, basic_metrics, effort_aware
import optimizers as OP

SEED = 42
N_WOLVES, N_ITERS = 20, 30
BUDGET = 1200        # generous cap: no method is truncated, all are counted identically


# -------------------------------------------------------------- Stage 1: grouping
def feature_groups(X, y, tau=0.85, rng=None):
    """Correlation union-find grouping + high-MI singletons -> candidate pool C."""
    d = X.shape[1]
    R = np.corrcoef(X, rowvar=False)
    R = np.where(np.isfinite(R), R, 0.0)
    parent = list(range(d))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(d):
        for j in range(i + 1, d):
            if abs(R[i, j]) > tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    mi = mutual_info_classif(X, y, random_state=SEED)
    groups = {}
    for i in range(d):
        groups.setdefault(find(i), []).append(i)
    reps = [max(g, key=lambda i: mi[i]) for g in groups.values()]
    med = float(np.median(mi))
    cand = sorted(set(reps) | {i for i in range(d) if mi[i] > med})
    if len(cand) < 3:
        cand = sorted(set(cand) | set(np.argsort(-mi)[:3].tolist()))
    return np.array(cand, dtype=int)


# -------------------------------------------------------------- fitness (Eq. 8)
def _fast_f1_auc_mcc(yb, p, thr=0.5):
    """Exact F1 / ROC-AUC / MCC without sklearn's per-call validation overhead."""
    yp = (p >= thr).astype(np.int8)
    pos = yb == 1
    tp = float(np.count_nonzero(yp[pos] == 1))
    fp = float(np.count_nonzero(yp[~pos] == 1))
    fn = float(np.count_nonzero(yp[pos] == 0))
    tn = float(np.count_nonzero(yp[~pos] == 0))
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den > 0 else 0.0
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        auc = 0.5
    else:
        order = np.argsort(p, kind="mergesort")
        ps = p[order]
        ranks = np.empty(len(p), float)
        i = 0
        while i < len(ps):                      # mid-ranks for ties, as in sklearn
            j = i
            while j + 1 < len(ps) and ps[j + 1] == ps[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        auc = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return f1, float(auc), float(mcc)


def make_fitness(Xtr, ytr, cand, weights=(0.4, 0.4, 0.2), beta=0.008, seed=SEED):
    """Composite F1/AUC/MCC fitness on a held-out 20% slice, L1-logistic probe (Eq. 8)."""
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.2,
                                      stratify=ytr, random_state=seed)
    yb = np.asarray(yb)
    nc = len(cand)
    w1, w2, w3 = weights
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                             max_iter=200, class_weight="balanced")

    def fit(mask):
        cols = cand[mask]
        if len(cols) == 0:
            return 1.0
        try:
            clf.fit(Xa[:, cols], ya)
            p = clf.decision_function(Xb[:, cols])
            p = 1.0 / (1.0 + np.exp(-np.clip(p, -60, 60)))
            f1, auc, mcc = _fast_f1_auc_mcc(yb, p)
        except Exception:
            return 1.0
        return (1.0 - (w1 * f1 + w2 * auc + w3 * (mcc + 1) / 2)) + beta * (len(cols) / nc)

    return fit


# -------------------------------------------------------------- filter baselines
def filter_select(kind, Xtr, ytr, k):
    d = Xtr.shape[1]
    k = max(1, min(k, d))
    if kind == "IG":
        s = mutual_info_classif(Xtr, ytr, random_state=SEED)
    elif kind == "ChiSquare":
        s, _ = chi2(np.clip(Xtr, 0, None), ytr)
        s = np.where(np.isfinite(s), s, 0)
    else:                                   # PCA loading ranking
        p = PCA(n_components=min(d, max(1, k)), random_state=SEED).fit(Xtr)
        s = np.abs(p.components_).T @ p.explained_variance_ratio_
    idx = np.argsort(-s)[:k]
    m = np.zeros(d, bool)
    m[idx] = True
    return m


# -------------------------------------------------------------- method registry
WRAPPERS = {
    "GA":        lambda nc, ev, rng: OP.ga(nc, ev, rng, N_WOLVES, N_ITERS),
    "PSO":       lambda nc, ev, rng: OP.bpso(nc, ev, rng, N_WOLVES, N_ITERS),
    "GWO":       lambda nc, ev, rng: OP.bgwo2(nc, ev, rng, N_WOLVES, N_ITERS),
    "EMWS":      lambda nc, ev, rng: OP.emws(nc, ev, rng, N_WOLVES, N_ITERS),
    # --- reviewer 1, comment 1: recent GWO variants and other established swarms
    "bGWO1":     lambda nc, ev, rng: OP.bgwo1(nc, ev, rng, N_WOLVES, N_ITERS),
    "OBCGWO":    lambda nc, ev, rng: OP.bgwo2(nc, ev, rng, N_WOLVES, N_ITERS, opposition=True),
    "SA-bGWO":   lambda nc, ev, rng: OP.bgwo2(nc, ev, rng, N_WOLVES, N_ITERS, anneal=True),
    "SR-GWO":    lambda nc, ev, rng: OP.bgwo2(nc, ev, rng, N_WOLVES, N_ITERS, repulse=True),
    "bWOA":      lambda nc, ev, rng: OP.bwoa(nc, ev, rng, N_WOLVES, N_ITERS),
    "bSSA":      lambda nc, ev, rng: OP.bssa(nc, ev, rng, N_WOLVES, N_ITERS),
    "BHO":       lambda nc, ev, rng: OP.bho(nc, ev, rng, N_WOLVES, N_ITERS),
    # --- reviewer 2 (revision 3): the binary hippopotamus selector recommended to us
    "Q2HO":      lambda nc, ev, rng: OP.q2ho(nc, ev, rng, N_WOLVES, N_ITERS),
}

ABLATIONS = {
    "DF-GWO":            dict(),
    "DF-GWO w/o group":  dict(_nogroup=True),
    "DF-GWO w/o chaos":  dict(chaos=False),
    "DF-GWO w/o Levy":   dict(levy_on=False),
    "DF-GWO w/o OBL":    dict(obl=False),
    "DF-GWO w/o restart": dict(restart=False),
    "DF-GWO w/o LS":     dict(localsearch=False),
    "DF-GWO chaos only": dict(levy_on=False, obl=False, restart=False, localsearch=False),
    "DF-GWO Levy only":  dict(chaos=False, obl=False, restart=False, localsearch=False),
    "DF-GWO OBL only":   dict(chaos=False, levy_on=False, restart=False, localsearch=False),
    "DF-GWO restart only": dict(chaos=False, levy_on=False, obl=False, localsearch=False),
    "DF-GWO LS only":    dict(chaos=False, levy_on=False, obl=False, restart=False),
    "DF-GWO base":       dict(chaos=False, levy_on=False, obl=False, restart=False,
                              localsearch=False),
}


def score_subset(Xtr, ytr, Xte, yte, locte, cols, seed=SEED):
    """Final Random Forest scorer + effort-aware metrics on the untouched test fold."""
    if len(cols) == 0:
        cols = np.arange(Xtr.shape[1])
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=seed, n_jobs=1)
    rf.fit(Xtr[:, cols], ytr)
    p = rf.predict_proba(Xte[:, cols])[:, 1]
    yp = rf.predict(Xte[:, cols])
    m = basic_metrics(yte, yp, p)
    m.update(effort_aware(yte, p, locte))
    m["nfeat"] = int(len(cols))
    return m


def run_dataset(key, methods, tau=0.85, weights=(0.4, 0.4, 0.2), beta=0.008,
                chaos_lo=0.5, seed=SEED, folds=5, curves=False):
    """Run the requested methods on one dataset under the shared 5-fold protocol."""
    X, y, names, loc, repo = load_dataset(key)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = {m: [] for m in methods}
    curve_store = {m: [] for m in methods} if curves else None
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        Xtr0, Xte0, keep = fold_preprocess(X[tr], X[te])
        ytr0, yte = y[tr], y[te]
        Xtr, ytr = smote(Xtr0, ytr0, rng=np.random.default_rng(seed + fi))
        d = Xtr.shape[1]
        cand_all = np.arange(d)
        cand_grp = feature_groups(Xtr, ytr, tau=tau)
        for mname in methods:
            # crc32, not hash(): Python randomises string hashes per process, so the
            # previous derivation made per-method seeds irreproducible across runs
            rng = np.random.default_rng(seed * 1000 + fi * 17
                                        + zlib.crc32(mname.encode()) % 9973)
            t0 = __import__("time").perf_counter()
            nev = nun = 0
            if mname == "AllFeatures":
                cols = cand_all
            elif mname in ("IG", "ChiSquare", "PCA"):
                cols = np.where(filter_select(mname, Xtr, ytr, d // 2))[0]
            elif mname in WRAPPERS:
                fitfn = make_fitness(Xtr, ytr, cand_all, weights, beta, seed + fi)
                ev = OP.Budget(fitfn, BUDGET)
                mask, _ = WRAPPERS[mname](len(cand_all), ev, rng)
                mask = ev.bestmask if ev.bestmask is not None else mask
                cols = cand_all[mask]
                nev, nun = ev.n, ev.n_unique
                if curves:
                    curve_store[mname].append(ev.curve(N_ITERS))
            elif mname in ABLATIONS:
                kw = dict(ABLATIONS[mname])
                cand = cand_all if kw.pop("_nogroup", False) else cand_grp
                fitfn = make_fitness(Xtr, ytr, cand, weights, beta, seed + fi)
                ev = OP.Budget(fitfn, BUDGET)
                mask, _ = OP.df_gwo(len(cand), ev, rng, N_WOLVES, N_ITERS,
                                    chaos_lo=chaos_lo, **kw)
                mask = ev.bestmask if ev.bestmask is not None else mask
                cols = cand[mask]
                nev, nun = ev.n, ev.n_unique
                if curves:
                    curve_store[mname].append(ev.curve(N_ITERS))
            else:
                raise KeyError(mname)
            el = __import__("time").perf_counter() - t0
            r = score_subset(Xtr, ytr, Xte0, yte, loc[te], cols, seed)
            r["time"] = el
            r["evals"] = nev
            r["uniq"] = nun
            r["poolsize"] = int(len(cand_grp) if mname in ABLATIONS
                                and not ABLATIONS[mname].get("_nogroup") else d)
            out[mname].append(r)
    agg = {}
    for m, rows in out.items():
        agg[m] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        agg[m]["nfeat_std"] = float(np.std([r["nfeat"] for r in rows]))
        agg[m]["auc_std"] = float(np.std([r["auc"] for r in rows]))
    if curves:
        return agg, {m: np.mean(v, axis=0).tolist() for m, v in curve_store.items() if v}
    return agg
