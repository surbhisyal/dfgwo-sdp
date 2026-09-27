"""E2 (Reviewer 1, comment 2a-2c): parameter sensitivity for chaos offset, beta,
fitness weights and the correlation threshold tau."""
import os, json, time, itertools
import numpy as np
from joblib import Parallel, delayed
from common import DATASETS
from pipeline import run_dataset

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

GRID = []
for v in [0.0, 0.25, 0.50, 0.75]:
    GRID.append(("chaos_lo", v, dict(chaos_lo=v)))
for v in [0.0, 0.004, 0.008, 0.016, 0.032]:
    GRID.append(("beta", v, dict(beta=v)))
for v in [(0.50, 0.30, 0.20), (0.40, 0.40, 0.20), (1 / 3, 1 / 3, 1 / 3),
          (0.60, 0.20, 0.20), (0.30, 0.30, 0.40)]:
    GRID.append(("weights", "(%.2f, %.2f, %.2f)" % v, dict(weights=v)))
for v in [0.75, 0.80, 0.85, 0.90, 0.95]:
    GRID.append(("tau", v, dict(tau=v)))


def one(gi, key):
    fam, val, kw = GRID[gi]
    f = os.path.join(OUT, f"sens_{gi}_{key}.json")
    if os.path.exists(f):
        return "cached"
    agg = run_dataset(key, ["DF-GWO"], **kw)
    json.dump({"family": fam, "value": val, "dataset": key, "agg": agg["DF-GWO"]},
              open(f, "w"), indent=1)
    return "ok"


if __name__ == "__main__":
    keys = [d[0] for d in DATASETS]
    jobs = [(gi, k) for gi in range(len(GRID)) for k in keys]
    t = time.perf_counter()
    Parallel(n_jobs=7, verbose=5)(delayed(one)(gi, k) for gi, k in jobs)
    print("DONE SENS", time.perf_counter() - t)
