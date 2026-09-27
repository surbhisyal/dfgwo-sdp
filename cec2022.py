"""CEC-2022 bound-constrained benchmark suite (Reviewer 1, comment 1).

The twelve functions of the CEC-2022 special session on single-objective bound
constrained numerical optimisation. Official shift vectors, rotation matrices and
shuffle indices are supplied by `opfunu` (>= 1.0.4), which ports the reference C/MATLAB
implementation; no transformation data is regenerated here, so the numbers produced by
this module are directly comparable with published CEC-2022 results.

Search range is [-100, 100]^D for every function; D in {10, 20}.

    F1            unimodal      shifted, rotated Zakharov
    F2 - F5       basic         Rosenbrock / Schaffer F7 / non-continuous Rastrigin / Levy
    F6 - F8       hybrid        3, 6 and 5 sub-components
    F9 - F12      composition   5, 3, 5 and 6 sub-functions
"""
import numpy as np

from opfunu.cec_based import cec2022 as _c

LO, HI = -100.0, 100.0

# (index, class label, known global optimum F*)
_SPEC = [
    (1,  "Unimodal",    300.0),
    (2,  "Basic",       400.0),
    (3,  "Basic",       600.0),
    (4,  "Basic",       800.0),
    (5,  "Basic",       900.0),
    (6,  "Hybrid",     1800.0),
    (7,  "Hybrid",     2000.0),
    (8,  "Hybrid",     2200.0),
    (9,  "Composition", 2300.0),
    (10, "Composition", 2400.0),
    (11, "Composition", 2600.0),
    (12, "Composition", 2700.0),
]

CLASS_ORDER = ["Unimodal", "Basic", "Hybrid", "Composition"]


def suite(dim):
    """[(name, callable, lo, hi, dim, class, f_star), ...] for the given dimension."""
    out = []
    for idx, cls, fstar in _SPEC:
        obj = getattr(_c, f"F{idx}2022")(ndim=dim)
        assert abs(obj.f_global - fstar) < 1e-9, (idx, obj.f_global)
        out.append((f"CEC22-F{idx}", obj.evaluate, LO, HI, dim, cls, fstar))
    return out


def error(value, fstar):
    """Solution error F(x) - F*, the quantity the CEC protocol reports."""
    return max(0.0, float(value) - float(fstar))


if __name__ == "__main__":
    for d in (10, 20):
        for name, f, lo, hi, dim, cls, fstar in suite(d):
            x = np.zeros(dim)
            print(f"{name:11s} D={dim:2d} {cls:12s} F*={fstar:7.1f}  f(0)={f(x):.6e}")
