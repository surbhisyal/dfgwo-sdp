"""Continuous optimisers under a strict, shared function-evaluation budget.

Reviewer 1 (comments 1, 2 and 4). Two things changed relative to the previous revision.

1.  Fairness. Every algorithm now stops at exactly the same number of objective-function
    evaluations (MAX_FE), not after the same number of iterations. Iteration-count parity
    is not parity: I-GWO scores two candidates per wolf per iteration, so under the old
    protocol it would silently receive twice the budget of GWO. The `FE` counter raises
    `BudgetExhausted` on the evaluation that would exceed the cap, and every algorithm
    below tolerates that exception at any point in its loop.

2.  Annealing on a common scale. Control schedules (the GWO `a`, the PSO inertia, ...)
    are indexed by the fraction of the budget consumed, p = FE / MAX_FE, rather than by
    iteration index. With unequal evaluations per iteration these are not the same thing,
    and p is the one that makes the comparison meaningful.

Convergence curves are recorded as best-so-far against evaluation count on a fixed grid,
so curves from algorithms with different internal structure are directly comparable.
"""
import numpy as np
from math import pi, sin, cos, gamma


class BudgetExhausted(Exception):
    pass


class FE:
    """Objective wrapper: counts evaluations, tracks the incumbent, records the curve."""

    def __init__(self, f, max_fe, npoints=200):
        self.f, self.max_fe = f, max_fe
        self.n = 0
        self.best, self.bx = np.inf, None
        self.step = max(1, max_fe // npoints)
        self.curve = []

    def __call__(self, x):
        if self.n >= self.max_fe:
            raise BudgetExhausted
        v = float(self.f(np.asarray(x, float)))
        if not np.isfinite(v):
            v = np.inf
        self.n += 1
        if v < self.best:
            self.best, self.bx = v, np.array(x, float)
        if self.n % self.step == 0:
            self.curve.append(self.best)
        return v

    @property
    def p(self):
        """Fraction of the evaluation budget consumed, in [0, 1]."""
        return self.n / self.max_fe

    def finish(self):
        want = self.max_fe // self.step
        while len(self.curve) < want:
            self.curve.append(self.best)
        return self.best, self.bx, self.curve[:want]


def _clip(x, lb, ub):
    return np.clip(x, lb, ub)


def _init(ev, lb, ub, dim, N, rng):
    X = rng.uniform(lb, ub, (N, dim))
    fit = np.array([ev(x) for x in X])
    return X, fit


def levy(dim, rng, beta=1.5):
    s = (gamma(1 + beta) * sin(pi * beta / 2) /
         (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, s, dim)
    v = rng.normal(0, 1, dim)
    return u / np.abs(v) ** (1 / beta)


def _leaders(X, fit):
    o = np.argsort(fit)
    return X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy(), o


# ============================================================ GWO and its variants
def c_gwo(ev, lb, ub, dim, N, rng):
    """Standard Grey Wolf Optimizer (Mirjalili et al. 2014)."""
    X, fit = _init(ev, lb, ub, dim, N, rng)
    A_, B_, D_, _ = _leaders(X, fit)
    while True:
        a = 2.0 * (1.0 - ev.p)
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            X[i] = _clip(np.mean(Xs, axis=0), lb, ub)
            fit[i] = ev(X[i])
        A_, B_, D_, _ = _leaders(X, fit)


def c_mgwo(ev, lb, ub, dim, N, rng):
    """mGWO (Mittal et al. 2016): non-linear decay a = 2(1 - p^2) in place of 2(1 - p),
    holding the exploration coefficient above 1 for the first ~71% of the budget."""
    X, fit = _init(ev, lb, ub, dim, N, rng)
    A_, B_, D_, _ = _leaders(X, fit)
    while True:
        a = 2.0 * (1.0 - ev.p ** 2)
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            X[i] = _clip(np.mean(Xs, axis=0), lb, ub)
            fit[i] = ev(X[i])
        A_, B_, D_, _ = _leaders(X, fit)


def c_cgwo(ev, lb, ub, dim, N, rng):
    """CGWO (Kohli & Arora 2018): the control parameter is driven by a chaotic map rather
    than a deterministic ramp. The Singer map is used, the best performer in their study.
    Distinct from DF-GWO, which perturbs the C coefficient with a logistic map."""
    X, fit = _init(ev, lb, ub, dim, N, rng)
    A_, B_, D_, _ = _leaders(X, fit)
    z = rng.uniform(0.05, 0.95)
    mu = 1.07
    while True:
        z = mu * (7.86 * z - 23.31 * z ** 2 + 28.75 * z ** 3 - 13.302875 * z ** 4)
        z = min(max(z, 1e-6), 1.0 - 1e-6)
        a = 2.0 * (1.0 - ev.p) * 2.0 * z
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            X[i] = _clip(np.mean(Xs, axis=0), lb, ub)
            fit[i] = ev(X[i])
        A_, B_, D_, _ = _leaders(X, fit)


def c_rwgwo(ev, lb, ub, dim, N, rng):
    """RW-GWO (Gupta & Deep 2019): the three leaders take Cauchy random walks of scale a
    and are accepted greedily; the omegas then follow the walked leaders."""
    X, fit = _init(ev, lb, ub, dim, N, rng)
    A_, B_, D_, o = _leaders(X, fit)
    fA, fB, fD = fit[o[0]], fit[o[1]], fit[o[2]]
    while True:
        a = 2.0 * (1.0 - ev.p)
        newL = []
        for L, fL in ((A_, fA), (B_, fB), (D_, fD)):
            Y = _clip(L + a * rng.standard_cauchy(dim), lb, ub)
            fY = ev(Y)
            newL.append((Y, fY) if fY < fL else (L, fL))
        (A_, fA), (B_, fB), (D_, fD) = newL
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            X[i] = _clip(np.mean(Xs, axis=0), lb, ub)
            fit[i] = ev(X[i])
        A_, B_, D_, o = _leaders(X, fit)
        fA, fB, fD = fit[o[0]], fit[o[1]], fit[o[2]]


def c_igwo(ev, lb, ub, dim, N, rng):
    """I-GWO (Nadimi-Shahraki et al. 2021): GWO plus dimension-learning-based hunting.
    Each wolf produces a GWO candidate and a DLH candidate built from a neighbour inside
    the radius ||X_i - X_GWO|| and a random pack member; the better of the two replaces
    the wolf only if it improves on it."""
    X, fit = _init(ev, lb, ub, dim, N, rng)
    A_, B_, D_, _ = _leaders(X, fit)
    while True:
        a = 2.0 * (1.0 - ev.p)
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            x_gwo = _clip(np.mean(Xs, axis=0), lb, ub)
            f_gwo = ev(x_gwo)

            Ri = np.linalg.norm(X[i] - x_gwo)
            d = np.linalg.norm(X - X[i], axis=1)
            nb = np.where(d <= Ri)[0]
            if len(nb) == 0:
                nb = np.array([rng.integers(N)])
            xn = X[nb[rng.integers(len(nb))]]
            xr = X[rng.integers(N)]
            x_dlh = _clip(X[i] + rng.random(dim) * (xn - xr), lb, ub)
            f_dlh = ev(x_dlh)

            cand, fc = (x_gwo, f_gwo) if f_gwo <= f_dlh else (x_dlh, f_dlh)
            if fc < fit[i]:
                X[i], fit[i] = cand, fc
        A_, B_, D_, _ = _leaders(X, fit)


def c_dfgwo(ev, lb, ub, dim, N, rng, chaos=True, levy_on=True, obl=True, restart=True,
            chaos_lo=0.5):
    """Continuous DF-GWO: cosine-annealed a, logistic-chaotic C, Levy perturbation,
    opposition-based initialisation and adaptive restart."""
    half = N // 2
    X = rng.uniform(lb, ub, (N, dim))
    if obl:
        X[half:] = lb + ub - X[:N - half]
    X = _clip(X, lb, ub)
    fit = np.array([ev(x) for x in X])
    A_, B_, D_, o = _leaders(X, fit)
    best, bx, stall = fit[o[0]], X[o[0]].copy(), 0
    z = rng.uniform(0.1, 0.9)
    while True:
        a = 2.0 * cos((pi / 2) * ev.p)
        z = 4.0 * z * (1.0 - z)
        for i in range(N):
            Xs = []
            for L in (A_, B_, D_):
                A = 2 * a * rng.random(dim) - a
                C = (2 * rng.random(dim) * (chaos_lo + (1 - chaos_lo) * z)) if chaos \
                    else 2 * rng.random(dim)
                Xs.append(L - A * np.abs(C * L - X[i]))
            xn = np.mean(Xs, axis=0)
            if levy_on and ev.p > 0.5 and rng.random() < 0.15:
                xn = xn + 0.01 * levy(dim, rng) * (xn - bx)
            xn = _clip(xn, lb, ub)
            fv = ev(xn)
            fit[i], X[i] = fv, xn
            if fv < best:
                best, bx, stall = fv, xn.copy(), 0
        A_, B_, D_, o = _leaders(X, fit)
        X[o[-1]], fit[o[-1]] = bx.copy(), best
        stall += 1
        if restart and stall >= 5:
            for w in o[-(N // 3):]:
                X[w] = rng.uniform(lb, ub, dim)
                fit[w] = ev(X[w])
            stall = 0


# ============================================================ hippopotamus family
def c_ho(ev, lb, ub, dim, N, rng):
    """Hippopotamus Optimization (Amiri et al. 2024)."""
    X, fit = _init(ev, lb, ub, dim, N, rng)
    g = int(np.argmin(fit))
    gb, best = X[g].copy(), fit[g]
    t = 0
    while True:
        t += 1
        for i in range(N):
            if i < N // 2:
                I = rng.integers(1, 3)
                y = X[i] + rng.random(dim) * (gb - I * X[i])
            elif rng.random() < 0.5:
                pred = rng.uniform(lb, ub, dim)
                dd = np.abs(pred - X[i]) + 1e-12
                y = np.where(fit[i] > best, pred + rng.random(dim) * 2 / dd,
                             X[i] + rng.random(dim) * (pred - X[i]))
            else:
                lo, hi = np.asarray(lb) / t, np.asarray(ub) / t
                y = X[i] + rng.random(dim) * (lo + rng.random(dim) * (hi - lo))
            y = _clip(y, lb, ub)
            fv = ev(y)
            if fv < fit[i]:
                fit[i], X[i] = fv, y
            if fv < best:
                best, gb = fv, y.copy()


def _quad_interp(x1, x2, x3, f1, f2, f3, lb, ub):
    """Dimension-wise quadratic-interpolation minimiser through three points."""
    num = (x2 ** 2 - x3 ** 2) * f1 + (x3 ** 2 - x1 ** 2) * f2 + (x1 ** 2 - x2 ** 2) * f3
    den = (x2 - x3) * f1 + (x3 - x1) * f2 + (x1 - x2) * f3
    with np.errstate(divide="ignore", invalid="ignore"):
        xq = 0.5 * num / den
    xq = np.where(np.isfinite(xq), xq, x1)
    return _clip(xq, lb, ub)


def c_sho(ev, lb, ub, dim, N, rng):
    """Strengthened HO (Saghafi et al. 2026): HO augmented with (i) quadratic
    interpolation through the incumbent and two random members, (ii) horizontal crossover
    between two individuals and vertical crossover between two dimensions of one
    individual, and (iii) randomised chaotic opposition-based learning on the worst
    third of the pack."""
    X, fit = _init(ev, lb, ub, dim, N, rng)
    g = int(np.argmin(fit))
    gb, best = X[g].copy(), fit[g]
    z = rng.uniform(0.1, 0.9)
    t = 0
    while True:
        t += 1
        # ---- HO core
        for i in range(N):
            if i < N // 2:
                I = rng.integers(1, 3)
                y = X[i] + rng.random(dim) * (gb - I * X[i])
            elif rng.random() < 0.5:
                pred = rng.uniform(lb, ub, dim)
                dd = np.abs(pred - X[i]) + 1e-12
                y = np.where(fit[i] > best, pred + rng.random(dim) * 2 / dd,
                             X[i] + rng.random(dim) * (pred - X[i]))
            else:
                lo, hi = np.asarray(lb) / t, np.asarray(ub) / t
                y = X[i] + rng.random(dim) * (lo + rng.random(dim) * (hi - lo))
            y = _clip(y, lb, ub)
            fv = ev(y)
            if fv < fit[i]:
                fit[i], X[i] = fv, y
            if fv < best:
                best, gb = fv, y.copy()

        # ---- quadratic interpolation around the incumbent
        j, k = rng.choice(N, 2, replace=False)
        xq = _quad_interp(gb, X[j], X[k], best, fit[j], fit[k], lb, ub)
        fq = ev(xq)
        if fq < best:
            best, gb = fq, xq.copy()
        w = int(np.argmax(fit))
        if fq < fit[w]:
            X[w], fit[w] = xq, fq

        # ---- horizontal crossover between two randomly paired individuals
        i1, i2 = rng.choice(N, 2, replace=False)
        r = rng.random(dim)
        ch = _clip(r * X[i1] + (1 - r) * X[i2]
                   + rng.uniform(-1, 1, dim) * (X[i1] - X[i2]), lb, ub)
        fc = ev(ch)
        if fc < fit[i1]:
            X[i1], fit[i1] = ch, fc
        if fc < best:
            best, gb = fc, ch.copy()

        # ---- vertical crossover between two dimensions of one individual
        if dim >= 2:
            i3 = rng.integers(N)
            d1, d2 = rng.choice(dim, 2, replace=False)
            cv = X[i3].copy()
            rr = rng.random()
            cv[d1] = rr * cv[d1] + (1 - rr) * cv[d2]
            cv = _clip(cv, lb, ub)
            fv2 = ev(cv)
            if fv2 < fit[i3]:
                X[i3], fit[i3] = cv, fv2
            if fv2 < best:
                best, gb = fv2, cv.copy()

        # ---- chaotic opposition-based learning on the worst third
        z = 4.0 * z * (1.0 - z)
        o = np.argsort(fit)
        for wI in o[-(N // 3):]:
            xo = _clip(np.asarray(lb) + np.asarray(ub) - z * X[wI], lb, ub)
            fo = ev(xo)
            if fo < fit[wI]:
                X[wI], fit[wI] = xo, fo
            if fo < best:
                best, gb = fo, xo.copy()


# ============================================================ classical comparators
def c_pso(ev, lb, ub, dim, N, rng):
    X, fit = _init(ev, lb, ub, dim, N, rng)
    V = np.zeros((N, dim))
    pb, pbf = X.copy(), fit.copy()
    g = int(np.argmin(fit))
    gb, best = X[g].copy(), fit[g]
    vmax = 0.2 * (np.asarray(ub) - np.asarray(lb))
    while True:
        w = 0.9 - 0.5 * ev.p
        V = np.clip(w * V + 2 * rng.random((N, dim)) * (pb - X)
                    + 2 * rng.random((N, dim)) * (gb - X), -vmax, vmax)
        X = _clip(X + V, lb, ub)
        for i in range(N):
            fit[i] = ev(X[i])
            if fit[i] < pbf[i]:
                pbf[i], pb[i] = fit[i], X[i].copy()
            if fit[i] < best:
                best, gb = fit[i], X[i].copy()


def c_woa(ev, lb, ub, dim, N, rng):
    X, fit = _init(ev, lb, ub, dim, N, rng)
    g = int(np.argmin(fit))
    gb, best = X[g].copy(), fit[g]
    while True:
        a = 2.0 * (1.0 - ev.p)
        for i in range(N):
            A = 2 * a * rng.random(dim) - a
            C = 2 * rng.random(dim)
            if rng.random() < 0.5:
                if np.mean(np.abs(A)) < 1:
                    xn = gb - A * np.abs(C * gb - X[i])
                else:
                    r = X[rng.integers(N)]
                    xn = r - A * np.abs(C * r - X[i])
            else:
                l = rng.uniform(-1, 1)
                xn = np.abs(gb - X[i]) * np.exp(l) * np.cos(2 * pi * l) + gb
            X[i] = _clip(xn, lb, ub)
            fit[i] = ev(X[i])
            if fit[i] < best:
                best, gb = fit[i], X[i].copy()


def c_sca(ev, lb, ub, dim, N, rng):
    X, fit = _init(ev, lb, ub, dim, N, rng)
    g = int(np.argmin(fit))
    gb, best = X[g].copy(), fit[g]
    while True:
        r1 = 2.0 * (1.0 - ev.p)
        for i in range(N):
            r2 = 2 * pi * rng.random(dim)
            r3 = 2 * rng.random(dim)
            r4 = rng.random(dim)
            step = np.where(r4 < 0.5, np.sin(r2), np.cos(r2))
            X[i] = _clip(X[i] + r1 * step * np.abs(r3 * gb - X[i]), lb, ub)
            fit[i] = ev(X[i])
            if fit[i] < best:
                best, gb = fit[i], X[i].copy()


def c_ga(ev, lb, ub, dim, N, rng):
    X, fit = _init(ev, lb, ub, dim, N, rng)
    span = np.asarray(ub) - np.asarray(lb)
    while True:
        o = np.argsort(fit)
        new = [X[o[0]].copy(), X[o[1]].copy()]
        newf = [fit[o[0]], fit[o[1]]]
        while len(new) < N:
            c = rng.choice(N, 3, False)
            p1 = X[c[np.argmin(fit[c])]]
            c = rng.choice(N, 3, False)
            p2 = X[c[np.argmin(fit[c])]]
            al = rng.random(dim)
            ch = al * p1 + (1 - al) * p2
            mm = rng.random(dim) < 0.1
            ch = np.where(mm, ch + 0.1 * span * rng.normal(0, 1, dim), ch)
            ch = _clip(ch, lb, ub)
            new.append(ch)
            newf.append(ev(ch))
        X = np.array(new[:N])
        fit = np.array(newf[:N])


# five GWO-family members, two hippopotamus-family members, four classical comparators
CONT = {
    "DF-GWO": c_dfgwo,
    "GWO": c_gwo, "I-GWO": c_igwo, "RW-GWO": c_rwgwo, "mGWO": c_mgwo, "CGWO": c_cgwo,
    "HO": c_ho, "SHO": c_sho,
    "PSO": c_pso, "WOA": c_woa, "SCA": c_sca, "GA": c_ga,
}

GWO_FAMILY = ["DF-GWO", "GWO", "I-GWO", "RW-GWO", "mGWO", "CGWO"]
HO_FAMILY = ["HO", "SHO"]


def run(alg, f, lb, ub, dim, max_fe, seed, npop=30):
    """One independent run. Returns (best value, best x, convergence curve)."""
    rng = np.random.default_rng(seed)
    ev = FE(f, max_fe)
    try:
        CONT[alg](ev, lb, ub, dim, npop, rng)
    except BudgetExhausted:
        pass
    return ev.finish()
