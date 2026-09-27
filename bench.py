"""Continuous benchmark suite (Reviewer 1, comment 3).

F1-F13: the classical unimodal / multimodal / fixed-dimension set of Yao et al. (1999)
used in the original GWO paper. Four constrained engineering-design problems supply the
real-world benchmark component.
"""
import numpy as np
from math import pi, cos, sin, exp, sqrt, gamma


# ------------------------------------------------------------------ F1 - F13
def F1(x):  return np.sum(x ** 2)
def F2(x):  return np.sum(np.abs(x)) + np.prod(np.abs(x))
def F3(x):  return np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])
def F4(x):  return np.max(np.abs(x))
def F5(x):  return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
def F6(x):  return np.sum(np.floor(x + 0.5) ** 2)
def F7(x):  return np.sum(np.arange(1, len(x) + 1) * x ** 4) + np.random.random()
def F8(x):  return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
def F9(x):  return np.sum(x ** 2 - 10 * np.cos(2 * pi * x) + 10)


def F10(x):
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * pi * x)) / n) + 20 + np.e)


def F11(x):
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1


def _u(x, a, k, m):
    return np.sum(k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a))


def F12(x):
    n = len(x); y = 1 + (x + 1) / 4
    return (pi / n) * (10 * np.sin(pi * y[0]) ** 2
                       + np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(pi * y[1:]) ** 2))
                       + (y[-1] - 1) ** 2) + _u(x, 10, 100, 4)


def F13(x):
    return (0.1 * (np.sin(3 * pi * x[0]) ** 2
                   + np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * pi * x[1:]) ** 2))
                   + (x[-1] - 1) ** 2 * (1 + np.sin(2 * pi * x[-1]) ** 2))
            + _u(x, 5, 100, 4))


BENCH = [
    ("F1",  F1,  -100, 100, 30, "Unimodal",   0.0),
    ("F2",  F2,  -10, 10, 30, "Unimodal",     0.0),
    ("F3",  F3,  -100, 100, 30, "Unimodal",   0.0),
    ("F4",  F4,  -100, 100, 30, "Unimodal",   0.0),
    ("F5",  F5,  -30, 30, 30, "Unimodal",     0.0),
    ("F6",  F6,  -100, 100, 30, "Unimodal",   0.0),
    ("F7",  F7,  -1.28, 1.28, 30, "Unimodal", 0.0),
    ("F8",  F8,  -500, 500, 30, "Multimodal", -12569.5),
    ("F9",  F9,  -5.12, 5.12, 30, "Multimodal", 0.0),
    ("F10", F10, -32, 32, 30, "Multimodal",   0.0),
    ("F11", F11, -600, 600, 30, "Multimodal", 0.0),
    ("F12", F12, -50, 50, 30, "Multimodal",   0.0),
    ("F13", F13, -50, 50, 30, "Multimodal",   0.0),
]


# ------------------------------------------------- constrained engineering problems
def _pen(g, k=1e10):
    return k * np.sum(np.maximum(0.0, np.asarray(g)) ** 2)


def pressure_vessel(x):
    Ts, Th, R, L = x
    Ts = 0.0625 * round(Ts / 0.0625); Th = 0.0625 * round(Th / 0.0625)
    f = 0.6224 * Ts * R * L + 1.7781 * Th * R ** 2 + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R
    g = [-Ts + 0.0193 * R, -Th + 0.00954 * R,
         -pi * R ** 2 * L - (4 / 3) * pi * R ** 3 + 1296000, L - 240]
    return f + _pen(g)


def welded_beam(x):
    h, l, t, b = x
    P, L, E, G = 6000., 14., 30e6, 12e6
    f = 1.10471 * h ** 2 * l + 0.04811 * t * b * (14.0 + l)
    M = P * (L + l / 2); R = sqrt(l ** 2 / 4 + ((h + t) / 2) ** 2)
    J = 2 * (sqrt(2) * h * l * (l ** 2 / 12 + ((h + t) / 2) ** 2))
    tau1 = P / (sqrt(2) * h * l); tau2 = M * R / J
    tau = sqrt(tau1 ** 2 + 2 * tau1 * tau2 * l / (2 * R) + tau2 ** 2)
    sig = 6 * P * L / (b * t ** 2)
    dl = 4 * P * L ** 3 / (E * b * t ** 3)
    Pc = (4.013 * E * sqrt(t ** 2 * b ** 6 / 36) / L ** 2) * (1 - t / (2 * L) * sqrt(E / (4 * G)))
    g = [tau - 13600, sig - 30000, h - b, 0.10471 * h ** 2 + 0.04811 * t * b * (14 + l) - 5.0,
         0.125 - h, dl - 0.25, P - Pc]
    return f + _pen(g)


def spring(x):
    d, D, N = x
    f = (N + 2) * D * d ** 2
    g = [1 - D ** 3 * N / (71785 * d ** 4),
         (4 * D ** 2 - d * D) / (12566 * (D * d ** 3 - d ** 4)) + 1 / (5108 * d ** 2) - 1,
         1 - 140.45 * d / (D ** 2 * N),
         (D + d) / 1.5 - 1]
    return f + _pen(g)


def speed_reducer(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    x3 = round(x3)
    f = (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
         - 1.508 * x1 * (x6 ** 2 + x7 ** 2) + 7.4777 * (x6 ** 3 + x7 ** 3)
         + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
    g = [27 / (x1 * x2 ** 2 * x3) - 1, 397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1,
         1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1, 1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1,
         sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110 * x6 ** 3) - 1,
         sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85 * x7 ** 3) - 1,
         x2 * x3 / 40 - 1, 5 * x2 / x1 - 1, x1 / (12 * x2) - 1,
         (1.5 * x6 + 1.9) / x4 - 1, (1.1 * x7 + 1.9) / x5 - 1]
    return f + _pen(g)


ENGINEERING = [
    ("Pressure vessel",  pressure_vessel, np.array([0.0625, 0.0625, 10., 10.]),
     np.array([99 * .0625, 99 * .0625, 200., 200.]), 6059.714),
    ("Welded beam",      welded_beam, np.array([0.1, 0.1, 0.1, 0.1]),
     np.array([2., 10., 10., 2.]), 1.724852),
    ("Tension/compression spring", spring, np.array([0.05, 0.25, 2.0]),
     np.array([2.0, 1.3, 15.0]), 0.012665),
    ("Speed reducer",    speed_reducer, np.array([2.6, 0.7, 17., 7.3, 7.8, 2.9, 5.0]),
     np.array([3.6, 0.8, 28., 8.3, 8.3, 3.9, 5.5]), 2994.4245),
]


# --------------------------------------------------------- continuous optimizers
def _clip(X, lb, ub):
    return np.clip(X, lb, ub)


def levy_step(dim, rng, beta=1.5):
    s = (gamma(1 + beta) * sin(pi * beta / 2) /
         (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, s, dim); v = rng.normal(0, 1, dim)
    return u / np.abs(v) ** (1 / beta)


def c_gwo(f, lb, ub, dim, N, T, rng):
    X = _clip(rng.uniform(lb, ub, (N, dim)), lb, ub)
    fit = np.array([f(x) for x in X]); o = np.argsort(fit)
    A_, B_, D_ = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
    best = fit[o[0]]; curve = []
    for t in range(1, T + 1):
        a = 2 - 2 * t / T
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a; C = 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            X[i] = _clip(np.mean(Xs, axis=0), lb, ub)
            fit[i] = f(X[i])
        o = np.argsort(fit)
        A_, B_, D_ = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
        best = min(best, fit[o[0]]); curve.append(best)
    return best, curve


def c_dfgwo(f, lb, ub, dim, N, T, rng, chaos=True, levy_on=True, obl=True, restart=True,
            chaos_lo=0.5):
    """Continuous DF-GWO: cosine annealing + logistic-chaotic C + Levy + OBL + restart."""
    half = N // 2
    X = rng.uniform(lb, ub, (N, dim))
    if obl:
        X[half:] = lb + ub - X[:N - half]
    X = _clip(X, lb, ub)
    fit = np.array([f(x) for x in X]); o = np.argsort(fit)
    A_, B_, D_ = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
    best, bx, stall = fit[o[0]], X[o[0]].copy(), 0
    z = rng.uniform(0.1, 0.9); curve = []
    for t in range(1, T + 1):
        a = 2.0 * cos((pi / 2) * (t / T))
        z = 4.0 * z * (1.0 - z)
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a
                C = (2 * rng.random(dim) * (chaos_lo + (1 - chaos_lo) * z)) if chaos \
                    else 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            xn = np.mean(Xs, axis=0)
            if levy_on and t > T / 2 and rng.random() < 0.15:
                xn = xn + 0.01 * levy_step(dim, rng) * (xn - bx)
            xn = _clip(xn, lb, ub); fv = f(xn)
            fit[i], X[i] = fv, xn          # standard GWO position updating
            if fv < best:
                best, bx, stall = fv, xn.copy(), 0
        o = np.argsort(fit)
        A_, B_, D_ = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
        X[o[-1]], fit[o[-1]] = bx.copy(), best
        stall += 1
        if restart and stall >= 5:
            for w in o[-(N // 3):]:
                X[w] = rng.uniform(lb, ub, dim); fit[w] = f(X[w])
            stall = 0
        curve.append(best)
    return best, curve


def c_pso(f, lb, ub, dim, N, T, rng):
    X = rng.uniform(lb, ub, (N, dim)); V = np.zeros((N, dim))
    fit = np.array([f(x) for x in X]); pb, pbf = X.copy(), fit.copy()
    g = int(np.argmin(fit)); gb, best = X[g].copy(), fit[g]; curve = []
    vmax = 0.2 * (np.asarray(ub) - np.asarray(lb))
    for t in range(T):
        w = 0.9 - 0.5 * t / T
        V = np.clip(w * V + 2 * rng.random((N, dim)) * (pb - X)
                    + 2 * rng.random((N, dim)) * (gb - X), -vmax, vmax)
        X = _clip(X + V, lb, ub)
        fit = np.array([f(x) for x in X])
        imp = fit < pbf; pbf[imp], pb[imp] = fit[imp], X[imp]
        if pbf.min() < best:
            best = pbf.min(); gb = pb[int(np.argmin(pbf))].copy()
        curve.append(best)
    return best, curve


def c_woa(f, lb, ub, dim, N, T, rng):
    X = rng.uniform(lb, ub, (N, dim)); fit = np.array([f(x) for x in X])
    g = int(np.argmin(fit)); gb, best = X[g].copy(), fit[g]; curve = []
    for t in range(T):
        a = 2 - 2 * t / T
        for i in range(N):
            A = 2 * a * rng.random(dim) - a; C = 2 * rng.random(dim)
            if rng.random() < 0.5:
                if np.mean(np.abs(A)) < 1:
                    xn = gb - A * np.abs(C * gb - X[i])
                else:
                    r = X[rng.integers(N)]; xn = r - A * np.abs(C * r - X[i])
            else:
                l = rng.uniform(-1, 1)
                xn = np.abs(gb - X[i]) * np.exp(l) * np.cos(2 * pi * l) + gb
            X[i] = _clip(xn, lb, ub); fit[i] = f(X[i])
            if fit[i] < best:
                best, gb = fit[i], X[i].copy()
        curve.append(best)
    return best, curve


def c_sca(f, lb, ub, dim, N, T, rng):
    X = rng.uniform(lb, ub, (N, dim)); fit = np.array([f(x) for x in X])
    g = int(np.argmin(fit)); gb, best = X[g].copy(), fit[g]; curve = []
    for t in range(1, T + 1):
        r1 = 2 - 2 * t / T
        for i in range(N):
            r2 = 2 * pi * rng.random(dim); r3 = 2 * rng.random(dim); r4 = rng.random(dim)
            step = np.where(r4 < 0.5, np.sin(r2), np.cos(r2))
            X[i] = _clip(X[i] + r1 * step * np.abs(r3 * gb - X[i]), lb, ub)
            fit[i] = f(X[i])
            if fit[i] < best:
                best, gb = fit[i], X[i].copy()
        curve.append(best)
    return best, curve


def c_ho(f, lb, ub, dim, N, T, rng):
    """Hippopotamus Optimization (Amiri et al. 2024), continuous form."""
    X = rng.uniform(lb, ub, (N, dim)); fit = np.array([f(x) for x in X])
    g = int(np.argmin(fit)); gb, best = X[g].copy(), fit[g]; curve = []
    for t in range(1, T + 1):
        for i in range(N):
            if i < N // 2:
                I = rng.integers(1, 3)
                y = X[i] + rng.random(dim) * (gb - I * X[i])
            elif rng.random() < 0.5:
                pred = rng.uniform(lb, ub, dim)
                d = np.abs(pred - X[i]) + 1e-12
                y = np.where(f(X[i]) > best, pred + rng.random(dim) * 2 / d,
                             X[i] + rng.random(dim) * (pred - X[i]))
            else:
                lo, hi = np.asarray(lb) / t, np.asarray(ub) / t
                y = X[i] + rng.random(dim) * (lo + rng.random(dim) * (hi - lo))
            y = _clip(y, lb, ub); fv = f(y)
            if fv < fit[i]:
                fit[i], X[i] = fv, y
            if fv < best:
                best, gb = fv, y.copy()
        curve.append(best)
    return best, curve


def c_ga(f, lb, ub, dim, N, T, rng):
    X = rng.uniform(lb, ub, (N, dim)); fit = np.array([f(x) for x in X])
    best = fit.min(); curve = []
    span = np.asarray(ub) - np.asarray(lb)
    for t in range(T):
        o = np.argsort(fit); new = [X[o[0]].copy(), X[o[1]].copy()]
        while len(new) < N:
            c = rng.choice(N, 3, False); p1 = X[c[np.argmin(fit[c])]]
            c = rng.choice(N, 3, False); p2 = X[c[np.argmin(fit[c])]]
            al = rng.random(dim); ch = al * p1 + (1 - al) * p2
            mm = rng.random(dim) < 0.1
            ch = np.where(mm, ch + 0.1 * span * rng.normal(0, 1, dim), ch)
            new.append(_clip(ch, lb, ub))
        X = np.array(new[:N]); fit = np.array([f(x) for x in X])
        best = min(best, fit.min()); curve.append(best)
    return best, curve


CONT = {"DF-GWO": c_dfgwo, "GWO": c_gwo, "PSO": c_pso, "WOA": c_woa,
        "SCA": c_sca, "HO": c_ho, "GA": c_ga}
