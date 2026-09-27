"""E3 (Reviewer 1, comments 2d and 4): component-wise ablation of DF-GWO."""
import os, json, time
import numpy as np
from joblib import Parallel, delayed
from common import DATASETS
from pipeline import run_dataset, ABLATIONS

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
CONFIGS = list(ABLATIONS.keys())


def one(key):
    f = os.path.join(OUT, f"abl_{key}.json")
    if os.path.exists(f):
        return key, "cached"
    t = time.perf_counter()
    agg, curves = run_dataset(key, CONFIGS, curves=True)
    json.dump({"agg": agg, "curves": curves}, open(f, "w"), indent=1)
    return key, f"{time.perf_counter()-t:.0f}s"


if __name__ == "__main__":
    order = ["ML", "PDE", "JDT", "camel-1.6", "camel-1.4", "xalan-2.5", "ant-1.7",
             "LC", "jedit-4.3", "xerces-1.2", "zxing", "jedit-4.2", "EQ", "prop-6",
             "ivy-2.0", "ant-1.6", "poi-2.0", "synapse-1.2", "apache", "safe"]
    t = time.perf_counter()
    res = Parallel(n_jobs=7, verbose=10)(delayed(one)(k) for k in order)
    for k, s in res:
        print(k, s, flush=True)
    print("DONE ABL", time.perf_counter() - t)
