"""E4 (Reviewer 1, comment 3): classical benchmark functions + engineering design problems."""
import os, json, time
import numpy as np
from joblib import Parallel, delayed
from bench import BENCH, ENGINEERING, CONT, c_dfgwo

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
NRUN, NPOP, NITER = 30, 30, 500


def run_fun(name, fn, lo, hi, dim):
    res, curves = {}, {}
    for mname, alg in CONT.items():
        vals, cs = [], []
        for r in range(NRUN):
            rng = np.random.default_rng(1000 + r)
            b, c = alg(fn, lo, hi, dim, NPOP, NITER, rng)
            vals.append(float(b)); cs.append(c)
        res[mname] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                          best=float(np.min(vals)), vals=vals)
        curves[mname] = np.mean(cs, axis=0)[::10].tolist()
    return name, res, curves


def run_eng(name, fn, lb, ub, known):
    dim = len(lb); res = {}
    for mname, alg in CONT.items():
        vals = []
        for r in range(NRUN):
            rng = np.random.default_rng(2000 + r)
            b, _ = alg(fn, lb, ub, dim, NPOP, NITER, rng)
            vals.append(float(b))
        res[mname] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                          best=float(np.min(vals)), worst=float(np.max(vals)),
                          vals=vals)
    return name, res, known


if __name__ == "__main__":
    t = time.perf_counter()
    out = Parallel(n_jobs=7, verbose=10)(
        delayed(run_fun)(n, f, lo, hi, d) for n, f, lo, hi, d, _, _ in BENCH)
    json.dump({n: {"res": r, "curves": c} for n, r, c in out},
              open(os.path.join(OUT, "bench_functions.json"), "w"), indent=1)
    print("functions done", time.perf_counter() - t, flush=True)

    out2 = Parallel(n_jobs=4, verbose=10)(
        delayed(run_eng)(n, f, lb, ub, k) for n, f, lb, ub, k in ENGINEERING)
    json.dump({n: {"res": r, "known": k} for n, r, k in out2},
              open(os.path.join(OUT, "bench_engineering.json"), "w"), indent=1)
    print("DONE BENCH", time.perf_counter() - t)
