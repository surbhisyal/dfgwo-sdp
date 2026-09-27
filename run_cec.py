"""Continuous experiments for revision 3.

E5  CEC-2022 bound-constrained suite, D = 10 and D = 20   (Reviewer 1, comments 1 and 4)
E6  classical F1-F13 at D = 30, expanded algorithm set    (Reviewer 1, comment 2)
E7  four constrained engineering-design problems          (Reviewer 1, comments 2 and 3)

Protocol
--------
* population 30, 30 independent runs from fixed seeds, identical for every algorithm;
* a strict evaluation budget of 1000 * D on CEC-2022 (10,000 at D = 10 and 20,000 at
  D = 20) and 15,000 on F1-F13 and the engineering problems;
* the budget counts every objective evaluation, so algorithms that score more than one
  candidate per individual per iteration are not given a hidden advantage;
* CEC-2022 results are reported as the solution error F(x) - F*, the quantity the CEC
  protocol defines;
* engineering solutions are checked for feasibility after the run, from the returned
  design vector, rather than being assumed feasible.
"""
import os, json, time
import numpy as np
from joblib import Parallel, delayed

import cbench as CB
import cec2022 as C22
import engineering as ENG
from bench import BENCH

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

NRUN, NPOP = 30, 30
FE_CLASSIC = 15000
ALGS = list(CB.CONT.keys())


def _stats(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), best=float(np.min(v)),
                worst=float(np.max(v)), median=float(np.median(v)),
                vals=[float(x) for x in v])


# ------------------------------------------------------------------ CEC-2022
def _one_cec(name, f, lo, hi, dim, cls, fstar, max_fe):
    res, cur = {}, {}
    for alg in ALGS:
        vals, curves = [], []
        for r in range(NRUN):
            b, bx, c = CB.run(alg, f, lo, hi, dim, max_fe, seed=7000 + r, npop=NPOP)
            vals.append(C22.error(b, fstar))
            curves.append([C22.error(v, fstar) for v in c])
        res[alg] = _stats(vals)
        A = np.asarray(curves, float)
        cur[alg] = dict(median=np.median(A, axis=0).tolist(),
                        mean=A.mean(axis=0).tolist())
    return name, cls, fstar, res, cur


def cec_suite(dim):
    max_fe = 1000 * dim
    jobs = [delayed(_one_cec)(n, f, lo, hi, d, cls, fs, max_fe)
            for n, f, lo, hi, d, cls, fs in C22.suite(dim)]
    out = Parallel(n_jobs=6, verbose=10)(jobs)
    blob = {n: dict(cls=cls, fstar=fs, res=r, curves=c) for n, cls, fs, r, c in out}
    blob["_meta"] = dict(dim=dim, max_fe=max_fe, nrun=NRUN, npop=NPOP, algs=ALGS)
    json.dump(blob, open(os.path.join(OUT, f"cec2022_D{dim}.json"), "w"), indent=1)
    return blob


# ------------------------------------------------------------------ F1-F13
def _one_classic(name, f, lo, hi, dim, cls, opt):
    res, cur = {}, {}
    for alg in ALGS:
        vals, curves = [], []
        for r in range(NRUN):
            b, bx, c = CB.run(alg, f, lo, hi, dim, FE_CLASSIC, seed=1000 + r, npop=NPOP)
            vals.append(float(b))
            curves.append(c)
        res[alg] = _stats(vals)
        A = np.asarray(curves, float)
        cur[alg] = dict(median=np.median(A, axis=0).tolist(),
                        mean=A.mean(axis=0).tolist())
    return name, cls, opt, res, cur


def classic_suite():
    jobs = [delayed(_one_classic)(n, f, lo, hi, d, cls, opt)
            for n, f, lo, hi, d, cls, opt in BENCH]
    out = Parallel(n_jobs=6, verbose=10)(jobs)
    blob = {n: dict(cls=cls, opt=o, res=r, curves=c) for n, cls, o, r, c in out}
    blob["_meta"] = dict(dim=30, max_fe=FE_CLASSIC, nrun=NRUN, npop=NPOP, algs=ALGS)
    json.dump(blob, open(os.path.join(OUT, "bench_functions_v3.json"), "w"), indent=1)
    return blob


# ------------------------------------------------------------------ engineering
def _one_eng(name, fg, lb, ub, ref):
    dim = len(lb)
    F = ENG.penalised(fg)
    res = {}
    for alg in ALGS:
        vals, viols, feas = [], [], []
        for r in range(NRUN):
            b, bx, c = CB.run(alg, F, lb, ub, dim, FE_CLASSIC, seed=2000 + r, npop=NPOP)
            fraw, viol, ok = ENG.report(fg, bx)
            # an infeasible run is recorded at its true objective, with the violation
            vals.append(fraw if ok else float("inf"))
            viols.append(viol)
            feas.append(bool(ok))
        ok_vals = [v for v in vals if np.isfinite(v)]
        res[alg] = dict(
            n_feasible=int(sum(feas)),
            feas_rate=float(np.mean(feas)),
            best=float(np.min(ok_vals)) if ok_vals else float("nan"),
            mean=float(np.mean(ok_vals)) if ok_vals else float("nan"),
            worst=float(np.max(ok_vals)) if ok_vals else float("nan"),
            std=float(np.std(ok_vals)) if ok_vals else float("nan"),
            max_viol=float(np.max(viols)),
            vals=[float(v) for v in vals])
    return name, res, ref


def eng_suite():
    jobs = [delayed(_one_eng)(n, fg, lb, ub, ref) for n, fg, lb, ub, ref in ENG.PROBLEMS]
    out = Parallel(n_jobs=4, verbose=10)(jobs)
    blob = {n: dict(res=r, ref=ref) for n, r, ref in out}
    blob["_meta"] = dict(max_fe=FE_CLASSIC, nrun=NRUN, npop=NPOP, algs=ALGS,
                         penalty=ENG.R_PENALTY, feas_tol=ENG.FEAS_TOL)
    json.dump(blob, open(os.path.join(OUT, "bench_engineering_v3.json"), "w"), indent=1)
    return blob


if __name__ == "__main__":
    t0 = time.perf_counter()
    eng_suite()
    print("engineering done", time.perf_counter() - t0, flush=True)
    classic_suite()
    print("F1-F13 done", time.perf_counter() - t0, flush=True)
    for d in (10, 20):
        cec_suite(d)
        print(f"CEC-2022 D{d} done", time.perf_counter() - t0, flush=True)
    print("DONE", time.perf_counter() - t0)
