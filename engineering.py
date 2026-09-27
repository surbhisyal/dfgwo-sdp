"""Constrained engineering-design problems with explicit constraint handling.

Reviewer 1, comment 3 asked how the penalty is applied. Each problem below exposes the
raw objective f(x) and the constraint vector g(x) <= 0 separately, so the penalised
objective actually optimised is visible and the feasibility of every reported solution
can be checked after the fact rather than asserted.

Penalised objective (static, non-adaptive, identical for every algorithm):

    F(x) = f(x) + R * sum_j max(0, g_j(x))^2 ,      R = 1e10

Every g_j below is written in normalised (dimensionless) form, i.e. each inequality is
divided by its own characteristic scale so that g_j = 0.01 means "this constraint is
violated by one per cent of its own magnitude" for every j. Without that normalisation a
single static R cannot be scale-appropriate: the pressure-vessel volume constraint has a
natural magnitude of 1.3e6 while its thickness constraints have a magnitude of about 1,
so one shared coefficient would penalise the two by nine orders of magnitude differently.
A solution is declared feasible when max_j g_j(x) <= 1e-4.

Side constraints are handled separately by clipping to [lb, ub] inside each optimiser,
so box bounds never contribute to the penalty. Discrete and integer design variables
(pressure-vessel shell thicknesses, speed-reducer tooth count) are snapped to their
admissible grid inside the objective before f and g are evaluated, so every candidate
scored is a manufacturable design.
"""
import numpy as np
from math import pi, sqrt

R_PENALTY = 1e10
FEAS_TOL = 1e-4


# --------------------------------------------------------------------- problems
def pressure_vessel(x):
    Ts, Th, R, L = x
    Ts = 0.0625 * round(Ts / 0.0625)          # discrete: multiples of 0.0625 in
    Th = 0.0625 * round(Th / 0.0625)
    f = (0.6224 * Ts * R * L + 1.7781 * Th * R ** 2
         + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R)
    g = [(-Ts + 0.0193 * R) / 1.0,
         (-Th + 0.00954 * R) / 1.0,
         (-pi * R ** 2 * L - (4.0 / 3.0) * pi * R ** 3 + 1296000.0) / 1296000.0,
         (L - 240.0) / 240.0]
    return f, np.asarray(g, float)


def welded_beam(x):
    h, l, t, b = x
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    f = 1.10471 * h ** 2 * l + 0.04811 * t * b * (14.0 + l)
    M = P * (L + l / 2.0)
    Rr = sqrt(l ** 2 / 4.0 + ((h + t) / 2.0) ** 2)
    J = 2.0 * (sqrt(2) * h * l * (l ** 2 / 12.0 + ((h + t) / 2.0) ** 2))
    tau1 = P / (sqrt(2) * h * l)
    tau2 = M * Rr / J
    tau = sqrt(tau1 ** 2 + 2 * tau1 * tau2 * l / (2 * Rr) + tau2 ** 2)
    sig = 6.0 * P * L / (b * t ** 2)
    dl = 4.0 * P * L ** 3 / (E * b * t ** 3)
    Pc = (4.013 * E * sqrt(t ** 2 * b ** 6 / 36.0) / L ** 2) * \
         (1 - t / (2 * L) * sqrt(E / (4 * G)))
    g = [(tau - 13600.0) / 13600.0,
         (sig - 30000.0) / 30000.0,
         (h - b) / 1.0,
         (0.10471 * h ** 2 + 0.04811 * t * b * (14.0 + l) - 5.0) / 5.0,
         (0.125 - h) / 0.125,
         (dl - 0.25) / 0.25,
         (P - Pc) / P]
    return f, np.asarray(g, float)


def spring(x):
    d, D, N = x
    f = (N + 2.0) * D * d ** 2
    g = [1.0 - D ** 3 * N / (71785.0 * d ** 4),
         (4 * D ** 2 - d * D) / (12566.0 * (D * d ** 3 - d ** 4)) + 1.0 / (5108.0 * d ** 2) - 1.0,
         1.0 - 140.45 * d / (D ** 2 * N),
         (D + d) / 1.5 - 1.0]
    return f, np.asarray(g, float)


def speed_reducer(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    x3 = round(x3)                             # integer: number of teeth
    f = (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
         - 1.508 * x1 * (x6 ** 2 + x7 ** 2) + 7.4777 * (x6 ** 3 + x7 ** 3)
         + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
    g = [27.0 / (x1 * x2 ** 2 * x3) - 1.0,
         397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1.0,
         1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1.0,
         1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1.0,
         sqrt((745.0 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110.0 * x6 ** 3) - 1.0,
         sqrt((745.0 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85.0 * x7 ** 3) - 1.0,
         x2 * x3 / 40.0 - 1.0,
         5.0 * x2 / x1 - 1.0,
         x1 / (12.0 * x2) - 1.0,
         (1.5 * x6 + 1.9) / x4 - 1.0,
         (1.1 * x7 + 1.9) / x5 - 1.0]
    return f, np.asarray(g, float)


# name, (f,g) callable, lb, ub, best known optimum from the literature
PROBLEMS = [
    ("Pressure vessel", pressure_vessel,
     np.array([0.0625, 0.0625, 10.0, 10.0]),
     np.array([99 * 0.0625, 99 * 0.0625, 200.0, 200.0]), 6059.7143),
    ("Welded beam", welded_beam,
     np.array([0.1, 0.1, 0.1, 0.1]), np.array([2.0, 10.0, 10.0, 2.0]), 1.724852),
    ("Tension/compression spring", spring,
     np.array([0.05, 0.25, 2.0]), np.array([2.0, 1.3, 15.0]), 0.012665),
    ("Speed reducer", speed_reducer,
     np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0]),
     np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]), 2994.4245),
]


def penalised(fg, R=R_PENALTY):
    """Wrap an (f, g) problem into the single scalar actually minimised."""
    def F(x):
        f, g = fg(x)
        v = np.maximum(0.0, g)
        return f + R * float(np.sum(v ** 2))
    return F


def report(fg, x):
    """Raw objective, maximum constraint violation and feasibility of a solution."""
    f, g = fg(np.asarray(x, float))
    viol = float(np.max(np.maximum(0.0, g))) if len(g) else 0.0
    return float(f), viol, bool(viol <= FEAS_TOL)
