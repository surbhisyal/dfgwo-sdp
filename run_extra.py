"""Add a newly introduced binary selector to the main comparison (revision 3).

Reviewer 2 recommended the binary hippopotamus selector Q2HO-MFTV. Rather than only
citing it, we run it as a comparator. The fold protocol is fixed by SEED in pipeline.py,
so a method evaluated now sees exactly the same folds, the same preprocessing and the
same evaluation budget as the sixteen methods evaluated previously; the new column is
merged into the existing per-dataset result files.
"""
import os, json, time
import numpy as np
from joblib import Parallel, delayed

from common import DATASETS
from pipeline import run_dataset

OUT = os.path.join(os.path.dirname(__file__), "results")
NEW = ["Q2HO"]


def one(key):
    path = os.path.join(OUT, f"main_{key}.json")
    blob = json.load(open(path))
    todo = [m for m in NEW if m not in blob["agg"]]
    if not todo:
        return key, "cached"
    t = time.perf_counter()
    agg, curves = run_dataset(key, todo, curves=True)
    blob["agg"].update(agg)
    blob.setdefault("curves", {}).update(curves)
    json.dump(blob, open(path, "w"), indent=1)
    return key, f"{time.perf_counter() - t:.0f}s"


if __name__ == "__main__":
    keys = [d[0] for d in DATASETS]
    res = Parallel(n_jobs=3, verbose=10)(delayed(one)(k) for k in keys)
    for k, s in res:
        print(k, s, flush=True)
    print("DONE EXTRA")
