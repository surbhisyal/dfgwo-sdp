"""Shared data loading / preprocessing / metrics for the DF-GWO revision experiments."""
import os, numpy as np, pandas as pd

# Where the three dataset repositories live. Point DFGWO_DATA at a directory holding
# PROMISE/, AEEEM/ and Relink/ subdirectories, or drop them into ./data next to this file.
# The datasets are public but are not redistributed here; see the README for sources.
BASE = os.environ.get(
    "DFGWO_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
FALLBACK = BASE
_SUBDIR = {"PROMISE": "PROMISE", "AEEEM": "AEEEM", "ReLink": "Relink"}

# (key, filename, repository, loc-column)
DATASETS = [
    ("ant-1.6",    "ant-1.6.csv",     "PROMISE", "loc"),
    ("ant-1.7",    "ant-1.7.csv",     "PROMISE", "loc"),
    ("camel-1.4",  "camel-1.4.csv",   "PROMISE", "loc"),
    ("camel-1.6",  "camel-1.6.csv",   "PROMISE", "loc"),
    ("ivy-2.0",    "data_ivy-2.0.csv","PROMISE", "loc"),
    ("prop-6",     "data_prop-6.csv", "PROMISE", "loc"),
    ("jedit-4.2",  "jedit-4.2.csv",   "PROMISE", "loc"),
    ("jedit-4.3",  "jedit-4.3.csv",   "PROMISE", "loc"),
    ("poi-2.0",    "poi-2.0.csv",     "PROMISE", "loc"),
    ("synapse-1.2","synapse-1.2.csv", "PROMISE", "loc"),
    ("xalan-2.5",  "xalan-2.5.csv",   "PROMISE", "loc"),
    ("xerces-1.2", "xerces-1.2.csv",  "PROMISE", "loc"),
    ("EQ",         "EQ.CSV",          "AEEEM",   "ck_oo_numberOfLinesOfCode"),
    ("JDT",        "JDT.csv",         "AEEEM",   "ck_oo_numberOfLinesOfCode"),
    ("LC",         "LC.csv",          "AEEEM",   "ck_oo_numberOfLinesOfCode"),
    ("ML",         "ML.csv",          "AEEEM",   "ck_oo_numberOfLinesOfCode"),
    ("PDE",        "PDE.csv",         "AEEEM",   "ck_oo_numberOfLinesOfCode"),
    ("apache",     "apache.csv",      "ReLink",  "CountLineCode"),
    ("safe",       "safe.csv",        "ReLink",  "CountLineCode"),
    ("zxing",      "zxing.csv",       "ReLink",  "CountLineCode"),
]

_DROP_ALWAYS = {"name", "version", "id", "filename", "class"}


def load_dataset(key):
    rec = [d for d in DATASETS if d[0] == key][0]
    _, fn, repo, loccol = rec
    sub = os.path.join(BASE, _SUBDIR[repo])
    cands = [os.path.join(sub, fn), os.path.join(sub, fn.replace(".CSV", ".csv")),
             os.path.join(sub, fn.replace(".csv", ".CSV")), os.path.join(FALLBACK, fn)]
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        raise FileNotFoundError(
            f"dataset '{key}' ({fn}, {repo}) not found under {BASE!r}.\n"
            f"Set the DFGWO_DATA environment variable to a directory containing "
            f"PROMISE/, AEEEM/ and Relink/ subdirectories, or place them in ./data. "
            f"See the README for where to obtain the datasets."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    # identify label column
    lab = None
    for cand in ("bug", "isDefective", "defects", "Defective"):
        if cand in df.columns:
            lab = cand
            break
    if lab is None:                       # AEEEM: unnamed trailing buggy/clean column
        lab = df.columns[-1]
    y_raw = df[lab]
    if y_raw.dtype == object:
        y = (y_raw.astype(str).str.strip().str.lower().isin(["buggy", "true", "yes", "1"])).astype(int).values
    else:
        y = (pd.to_numeric(y_raw, errors="coerce").fillna(0).values > 0).astype(int)
    X = df.drop(columns=[lab])
    drop = [c for c in X.columns if c.strip().lower() in _DROP_ALWAYS]
    X = X.drop(columns=drop)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any(axis=0)]
    loc = X[loccol].astype(float).values if loccol in X.columns else np.ones(len(X))
    loc = np.where(np.isfinite(loc) & (loc > 0), loc, 1.0)
    return X.values.astype(float), y, list(X.columns), loc, repo


# ---------------------------------------------------------------- preprocessing
def fold_preprocess(Xtr, Xte):
    """Median impute + drop zero-variance + min-max scale, all fitted on the training fold."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    keep = Xtr.std(axis=0) > 1e-12
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (Xtr - lo) / rng, np.clip((Xte - lo) / rng, -0.5, 1.5), keep


def smote(X, y, k=5, rng=None):
    """Minority over-sampling to parity (Chawla et al. 2002). Training folds only."""
    rng = rng or np.random.default_rng(0)
    cls, cnt = np.unique(y, return_counts=True)
    if len(cls) < 2:
        return X, y
    minc = cls[np.argmin(cnt)]
    need = cnt.max() - cnt.min()
    Xm = X[y == minc]
    if need <= 0 or len(Xm) < 2:
        return X, y
    kk = min(k, len(Xm) - 1)
    d = np.linalg.norm(Xm[:, None, :] - Xm[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :kk]
    idx = rng.integers(0, len(Xm), need)
    pick = nn[idx, rng.integers(0, kk, need)]
    gap = rng.random((need, X.shape[1]))
    Xnew = Xm[idx] + gap * (Xm[pick] - Xm[idx])
    return np.vstack([X, Xnew]), np.concatenate([y, np.full(need, minc)])


# ---------------------------------------------------------------- metrics
def basic_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(acc=acc, f1=f1, auc=auc, mcc=mcc, gmean=float(np.sqrt(sens * spec)))


def effort_aware(y_true, y_prob, loc, cut=0.20):
    """Recall@20%LOC, Popt@20% and effort-aware AUC using defect-density ranking."""
    loc = np.asarray(loc, float); loc = np.where(loc > 0, loc, 1.0)
    dens = y_prob / loc
    order = np.argsort(-dens)
    L, Y = loc[order], y_true[order]
    cl = np.cumsum(L) / L.sum()
    cy = np.cumsum(Y) / max(Y.sum(), 1)
    k = np.searchsorted(cl, cut) + 1
    recall20 = cy[min(k, len(cy)) - 1]
    ea_auc = float(np.trapezoid(cy, cl)) if hasattr(np, "trapezoid") else float(np.trapz(cy, cl))

    def area(o, upto=cut):
        l = np.cumsum(loc[o]) / loc.sum(); c = np.cumsum(y_true[o]) / max(y_true.sum(), 1)
        m = l <= upto
        if m.sum() < 2:
            return 0.0
        return float(np.trapezoid(c[m], l[m])) if hasattr(np, "trapezoid") else float(np.trapz(c[m], l[m]))
    opt = np.argsort(-(y_true / loc)); wst = np.argsort(y_true / loc)
    a_m, a_o, a_w = area(order), area(opt), area(wst)
    popt = 1 - (a_o - a_m) / (a_o - a_w) if (a_o - a_w) > 1e-12 else 0.0
    return dict(recall20=float(recall20), popt20=float(np.clip(popt, 0, 1)), ea_auc=ea_auc)
