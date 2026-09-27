"""E0+E1: main comparison over 20 datasets, 16 methods, shared folds."""
import os, sys, json, time
import numpy as np
from joblib import Parallel, delayed
from common import DATASETS
from pipeline import run_dataset

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

METHODS = ["AllFeatures", "IG", "ChiSquare", "PCA",
           "GA", "PSO", "GWO", "EMWS",
           "bGWO1", "OBCGWO", "SA-bGWO", "SR-GWO", "bWOA", "bSSA", "BHO",
           "DF-GWO"]


def one(key):
    f = os.path.join(OUT, f"main_{key}.json")
    if os.path.exists(f):
        return key, "cached"
    t = time.perf_counter()
    agg, curves = run_dataset(key, METHODS, curves=True)
    json.dump({"agg": agg, "curves": curves}, open(f, "w"), indent=1)
    return key, f"{time.perf_counter()-t:.0f}s"


if __name__ == "__main__":
    keys = [d[0] for d in DATASETS]
    # longest datasets first so the parallel pool drains evenly
    order = ["ML", "PDE", "JDT", "camel-1.6", "camel-1.4", "xalan-2.5", "ant-1.7",
             "LC", "jedit-4.3", "xerces-1.2", "zxing", "jedit-4.2", "EQ", "prop-6",
             "ivy-2.0", "ant-1.6", "poi-2.0", "synapse-1.2", "apache", "safe"]
    keys = [k for k in order if k in keys] + [k for k in keys if k not in order]
    res = Parallel(n_jobs=7, verbose=10)(delayed(one)(k) for k in keys)
    for k, s in res:
        print(k, s, flush=True)
    print("DONE MAIN")
