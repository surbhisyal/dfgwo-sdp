"""Binary feature-selection optimizers used in the DF-GWO revision.

Every wrapper receives an identical evaluation budget (N x T + N evaluations)
and the same cached fitness probe, so comparisons are like-for-like.
"""
import numpy as np
from math import gamma, pi, cos, sin


# --------------------------------------------------------------------- helpers
def _repair(mask, rng, kmin=3):
    if mask.sum() >= kmin:
        return mask
    off = np.where(~mask)[0]
    if len(off) == 0:
        return mask
    add = rng.choice(off, size=min(kmin - int(mask.sum()), len(off)), replace=False)
    mask = mask.copy()
    mask[add] = True
    return mask


def _sig(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(k * x, -60, 60)))


def levy(size, beta=1.5, rng=None):
    rng = rng or np.random.default_rng()
    s = (gamma(1 + beta) * sin(pi * beta / 2) /
         (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, s, size)
    v = rng.normal(0, 1, size)
    return u / np.abs(v) ** (1 / beta)


class Budget:
    """Fitness wrapper enforcing an identical evaluation budget across methods.

    Every call counts against the budget, duplicates included, which is the standard
    definition of a function-evaluation budget; the cache only avoids refitting the
    probe and never buys a method extra evaluations.
    """

    def __init__(self, fn, maxeval):
        self.fn, self.maxeval, self.n = fn, maxeval, 0
        self.cache, self.best, self.bestmask, self.trace = {}, np.inf, None, []

    def __call__(self, mask):
        self.n += 1
        if self.n > self.maxeval:
            return self.best if np.isfinite(self.best) else 1.0
        key = np.packbits(mask).tobytes()
        if key in self.cache:
            v = self.cache[key]
        else:
            v = float(self.fn(mask))
            self.cache[key] = v
        if v < self.best:
            self.best, self.bestmask = v, mask.copy()
        self.trace.append(self.best)
        return v

    @property
    def n_unique(self):
        return len(self.cache)

    def curve(self, T):
        """Best-so-far trace resampled onto T points."""
        if not self.trace:
            return [self.best] * T
        idx = np.linspace(0, len(self.trace) - 1, T).astype(int)
        return list(np.array(self.trace)[idx])


# --------------------------------------------------------------------- DF-GWO
def df_gwo(nc, ev, rng, N=20, T=30, chaos=True, levy_on=True, obl=True,
           restart=True, localsearch=True, chaos_lo=0.5, kmin=3):
    """Proposed optimizer. The component switches expose the ablation configurations."""
    half = N // 2
    if obl:
        P = rng.random((half, nc)) < 0.5
        P = np.vstack([P, ~P])
    else:
        P = rng.random((N, nc)) < 0.5
    X = np.where(P, rng.uniform(0.5, 3.0, (N, nc)), rng.uniform(-3.0, -0.5, (N, nc)))
    P = np.array([_repair(p, rng, kmin) for p in P])
    f = np.array([ev(p) for p in P])
    o = np.argsort(f)
    a, b, d = P[o[0]].copy(), P[o[1]].copy(), P[o[2]].copy()
    Xa, Xb, Xd = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
    gbest, gf, stall = P[o[0]].copy(), f[o[0]], 0
    z = rng.uniform(0.1, 0.9)
    for t in range(1, T + 1):
        av = 2.0 * cos((pi / 2) * (t / T))          # cosine-annealed weight, Eq. (6)
        z = 4.0 * z * (1.0 - z)                      # logistic map, Eq. (4)
        ksteep = 1.0 + 4.0 * (t / T)                 # annealed sigmoid steepness
        for i in range(N):
            Xs = []
            for L, Lx in ((a, Xa), (b, Xb), (d, Xd)):
                A = 2 * av * rng.random(nc) - av     # Eq. (1)
                if chaos:
                    C = 2 * rng.random(nc) * (chaos_lo + (1 - chaos_lo) * z)   # Eq. (7)
                else:
                    C = 2 * rng.random(nc)
                Dv = np.abs(C * Lx - X[i])           # Eq. (2)
                Xs.append(Lx - A * Dv)
            xnew = np.mean(Xs, axis=0)
            m = rng.random(nc) < _sig(xnew, ksteep)  # Eq. (3), annealed
            if levy_on and t > T / 2 and rng.random() < 0.15:
                nf = int(min(nc, 1 + abs(levy(1, rng=rng)[0])))
                bits = rng.choice(nc, size=min(nf, nc), replace=False)
                m[bits] = ~m[bits]
            m = _repair(m, rng, kmin)
            fv = ev(m)
            if fv < f[i]:
                f[i], P[i], X[i] = fv, m, xnew
            if fv < gf:
                gf, gbest, stall = fv, m.copy(), 0
        o = np.argsort(f)
        a, b, d = P[o[0]].copy(), P[o[1]].copy(), P[o[2]].copy()
        Xa, Xb, Xd = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
        P[o[-1]], f[o[-1]] = gbest.copy(), gf         # elitism injection
        stall += 1
        if restart and stall >= 5:                    # adaptive restart
            for w in o[-(N // 3):]:
                P[w] = _repair(rng.random(nc) < 0.5, rng, kmin)
                X[w] = rng.uniform(-3, 3, nc)
                f[w] = ev(P[w])
            stall = 0
    if localsearch:                                   # bounded single-bit-flip refinement
        for _ in range(2):
            improved = False
            for j in rng.permutation(nc):
                cand = gbest.copy()
                cand[j] = ~cand[j]
                if cand.sum() < kmin:
                    continue
                fv = ev(cand)
                if fv < gf - 1e-4:
                    gf, gbest, improved = fv, cand, True
            if not improved:
                break
    return gbest, gf


# ---------------------------------------------------------- GWO-family baselines
def bgwo1(nc, ev, rng, N=20, T=30, kmin=3):
    """Emary et al. (2016) bGWO1: stochastic crossover of three leader bit-vectors."""
    P = np.array([_repair(rng.random(nc) < 0.5, rng, kmin) for _ in range(N)])
    f = np.array([ev(p) for p in P])
    o = np.argsort(f)
    a, b, d = P[o[0]].copy(), P[o[1]].copy(), P[o[2]].copy()
    gb, gf = a.copy(), f[o[0]]
    for t in range(1, T + 1):
        av = 2 - 2 * t / T
        for i in range(N):
            bits = []
            for L in (a, b, d):
                A = 2 * av * rng.random(nc) - av
                bstep = (rng.random(nc) < _sig(10 * (A * 0.5 - 0.5))).astype(int)
                bits.append(np.bitwise_xor(L.astype(int), bstep))
            m = np.sum(bits, axis=0) >= 2
            m = _repair(m, rng, kmin)
            fv = ev(m)
            if fv < f[i]:
                f[i], P[i] = fv, m
            if fv < gf:
                gf, gb = fv, m.copy()
        o = np.argsort(f)
        a, b, d = P[o[0]].copy(), P[o[1]].copy(), P[o[2]].copy()
    return gb, gf


def bgwo2(nc, ev, rng, N=20, T=30, kmin=3, opposition=False, anneal=False,
          repulse=False):
    """Emary bGWO2 (sigmoid transfer). Flags give the OBCGWO / SA-bGWO / SR-GWO variants."""
    P = np.array([_repair(rng.random(nc) < 0.5, rng, kmin) for _ in range(N)])
    X = np.where(P, rng.uniform(0.5, 3, (N, nc)), rng.uniform(-3, -0.5, (N, nc)))
    f = np.array([ev(p) for p in P])
    o = np.argsort(f)
    a, b, d = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
    gb, gf = P[o[0]].copy(), f[o[0]]
    Temp = 1.0
    for t in range(1, T + 1):
        av = 2 - 2 * t / T
        if anneal:
            Temp = max(0.02, 0.95 ** t)
        for i in range(N):
            Xs = []
            for L in (a, b, d):
                A = 2 * av * rng.random(nc) - av
                C = 2 * rng.random(nc)
                Xs.append(L - A * np.abs(C * L - X[i]))
            xn = np.mean(Xs, axis=0)
            if repulse:                               # Wang et al. (2025) self-repulsion
                xn = xn + 0.1 * av * (X[i] - np.mean([a, b, d], axis=0))
            m = _repair(rng.random(nc) < _sig(xn), rng, kmin)
            fv = ev(m)
            acc = fv < f[i] or (anneal and rng.random() < np.exp(-(fv - f[i]) / Temp))
            if acc:
                f[i], P[i], X[i] = fv, m, xn
            if fv < gf:
                gf, gb = fv, m.copy()
        if opposition:                                # Too & Abdullah (2021)
            o = np.argsort(f)
            for w in o[-(N // 2):]:
                m = _repair(~P[w], rng, kmin)
                fv = ev(m)
                if fv < f[w]:
                    f[w], P[w], X[w] = fv, m, -X[w]
                if fv < gf:
                    gf, gb = fv, m.copy()
        o = np.argsort(f)
        a, b, d = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
    return gb, gf


# --------------------------------------------------------- other swarm / EC methods
def bpso(nc, ev, rng, N=20, T=30, kmin=3):
    X = rng.uniform(-3, 3, (N, nc))
    V = rng.uniform(-1, 1, (N, nc))
    P = np.array([_repair(rng.random(nc) < _sig(x), rng, kmin) for x in X])
    f = np.array([ev(p) for p in P])
    pb, pbf = P.copy(), f.copy()
    g = int(np.argmin(f))
    gb, gf, Xg = P[g].copy(), f[g], X[g].copy()
    for t in range(T):
        w = 0.9 - 0.5 * t / T
        V = (w * V
             + 2 * rng.random((N, nc)) * (np.where(pb, 1.5, -1.5) - X)
             + 2 * rng.random((N, nc)) * (Xg - X))
        V = np.clip(V, -4, 4)
        X = np.clip(X + V, -6, 6)
        for i in range(N):
            m = _repair(rng.random(nc) < _sig(X[i]), rng, kmin)
            fv = ev(m)
            if fv < pbf[i]:
                pbf[i], pb[i] = fv, m
            if fv < gf:
                gf, gb, Xg = fv, m.copy(), X[i].copy()
    return gb, gf


def ga(nc, ev, rng, N=20, T=30, kmin=3):
    P = np.array([_repair(rng.random(nc) < 0.5, rng, kmin) for _ in range(N)])
    f = np.array([ev(p) for p in P])
    for t in range(T):
        elite = np.argsort(f)[:2]
        new = [P[e].copy() for e in elite]
        while len(new) < N:
            c = rng.choice(N, 3, replace=False)
            p1 = P[c[np.argmin(f[c])]]
            c = rng.choice(N, 3, replace=False)
            p2 = P[c[np.argmin(f[c])]]
            pt = int(rng.integers(1, nc)) if nc > 1 else 1
            ch = np.concatenate([p1[:pt], p2[pt:]])
            mut = rng.random(nc) < (1.0 / nc)
            ch = np.where(mut, ~ch, ch)
            new.append(_repair(ch, rng, kmin))
        P = np.array(new[:N])
        f = np.array([ev(p) for p in P])
    i = int(np.argmin(f))
    return P[i], f[i]


def bwoa(nc, ev, rng, N=20, T=30, kmin=3):
    """Mafarja & Mirjalili (2018) binary whale optimisation."""
    X = rng.uniform(-3, 3, (N, nc))
    P = np.array([_repair(rng.random(nc) < _sig(x), rng, kmin) for x in X])
    f = np.array([ev(p) for p in P])
    g = int(np.argmin(f))
    gb, gf, Xg = P[g].copy(), f[g], X[g].copy()
    for t in range(T):
        av = 2 - 2 * t / T
        for i in range(N):
            A = 2 * av * rng.random(nc) - av
            C = 2 * rng.random(nc)
            if rng.random() < 0.5:
                if np.mean(np.abs(A)) < 1:
                    xn = Xg - A * np.abs(C * Xg - X[i])
                else:
                    r = X[rng.integers(N)]
                    xn = r - A * np.abs(C * r - X[i])
            else:
                l = rng.uniform(-1, 1)
                xn = np.abs(Xg - X[i]) * np.exp(l) * np.cos(2 * pi * l) + Xg
            X[i] = np.clip(xn, -6, 6)
            m = _repair(rng.random(nc) < _sig(X[i]), rng, kmin)
            fv = ev(m)
            if fv < f[i]:
                f[i], P[i] = fv, m
            if fv < gf:
                gf, gb, Xg = fv, m.copy(), X[i].copy()
    return gb, gf


def bssa(nc, ev, rng, N=20, T=30, kmin=3):
    """Faris et al. (2018) binary salp swarm."""
    X = rng.uniform(-3, 3, (N, nc))
    P = np.array([_repair(rng.random(nc) < _sig(x), rng, kmin) for x in X])
    f = np.array([ev(p) for p in P])
    g = int(np.argmin(f))
    gb, gf, Xg = P[g].copy(), f[g], X[g].copy()
    for t in range(1, T + 1):
        c1 = 2 * np.exp(-((4 * t / T) ** 2))
        for i in range(N):
            if i < N // 2:
                c2, c3 = rng.random(nc), rng.random(nc)
                X[i] = Xg + c1 * (6 * c2 - 3) * np.where(c3 < 0.5, 1, -1)
            else:
                X[i] = 0.5 * (X[i] + X[i - 1])
            X[i] = np.clip(X[i], -6, 6)
            m = _repair(rng.random(nc) < _sig(X[i]), rng, kmin)
            fv = ev(m)
            if fv < f[i]:
                f[i], P[i] = fv, m
            if fv < gf:
                gf, gb, Xg = fv, m.copy(), X[i].copy()
    return gb, gf


def bho(nc, ev, rng, N=20, T=30, kmin=3):
    """Binary Hippopotamus Optimization (Amiri et al. 2024) with the V-shaped
    transfer function used by the HO-MFTV binary selector."""
    X = rng.uniform(-3, 3, (N, nc))

    def tob(x):
        return np.abs(np.tanh(x)) > rng.random(nc)

    P = np.array([_repair(tob(x), rng, kmin) for x in X])
    f = np.array([ev(p) for p in P])
    g = int(np.argmin(f))
    gb, gf, Xg = P[g].copy(), f[g], X[g].copy()
    for t in range(1, T + 1):
        for i in range(N):
            if i < N // 2:                                   # phase 1: river herd
                I = rng.integers(1, 3)
                y = X[i] + rng.random(nc) * (Xg - I * X[i])
            elif rng.random() < 0.5:                         # phase 2: predator defence
                pred = rng.uniform(-3, 3, nc)
                dist = np.abs(pred - X[i]) + 1e-9
                y = np.where(f[i] > gf,
                             pred + (2 * rng.random(nc)) / dist,
                             X[i] + rng.random(nc) * (pred - X[i]))
            else:                                            # phase 3: local escape
                lo, hi = -3.0 / t, 3.0 / t
                y = X[i] + rng.random(nc) * (lo + rng.random(nc) * (hi - lo))
            y = np.clip(y, -6, 6)
            m = _repair(tob(y), rng, kmin)
            fv = ev(m)
            if fv < f[i]:
                f[i], P[i], X[i] = fv, m, y
            if fv < gf:
                gf, gb, Xg = fv, m.copy(), y.copy()
    return gb, gf


def emws(nc, ev, rng, N=20, T=30, kmin=3):
    """Zhu et al. (2021) EMWS: whale search followed by a simulated-annealing refinement."""
    gb, gf = bwoa(nc, ev, rng, N=N, T=int(T * 0.8), kmin=kmin)
    Temp = 0.15
    cur, cf = gb.copy(), gf
    for _ in range(max(1, int(T * 0.2)) * N):
        cand = cur.copy()
        j = int(rng.integers(nc))
        cand[j] = ~cand[j]
        cand = _repair(cand, rng, kmin)
        fv = ev(cand)
        if fv < cf or rng.random() < np.exp(-(fv - cf) / max(Temp, 1e-6)):
            cur, cf = cand, fv
        if fv < gf:
            gf, gb = fv, cand.copy()
        Temp *= 0.97
    return gb, gf


# --------------------------------------------------------------------- Q2HO-MFTV
# Transfer-function bank used by Q2HO-MFTV. tau is the time-varying slope: large early
# (gentle curve, many bits flip, exploration), small late (steep curve, few bits flip,
# exploitation). S-shaped rules set the bit from the transfer value; V-shaped rules
# complement the incumbent bit, which is what makes them behave differently.
def _tf_bank(x, tau, cur, rng):
    z = x / tau
    return [
        ("S1", 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))), "S"),
        ("S2", 1.0 / (1.0 + np.exp(-np.clip(2.0 * z, -60, 60))), "S"),
        ("V1", np.abs(np.tanh(z)), "V"),
        ("V2", np.abs(z) / np.sqrt(1.0 + z ** 2), "V"),
    ]


def _apply_tf(x, tau, cur, act, rng):
    name, val, kind = _tf_bank(x, tau, cur, rng)[act]
    r = rng.random(len(x))
    if kind == "S":
        return r < val
    return np.where(r < val, ~cur, cur)


def q2ho(nc, ev, rng, N=20, T=30, kmin=3, tau_hi=4.0, tau_lo=1.0,
         alpha=0.1, gamma_q=0.9, eps0=0.3):
    """Q2HO-MFTV (Mehrabi Hashjin et al. 2025): binary Hippopotamus Optimization with
    quantum-inspired chaotic initialisation, a bank of fuzzy time-varying transfer
    functions, and a tabular Q-learning controller that chooses among them online.

    State is the stagnation level of the global best (0 = improved last iteration,
    1 = stalled once or twice, 2 = stalled three times or more); the four actions are
    the four transfer functions; the reward is 1 when the iteration improves the
    incumbent and 0 otherwise. Budget accounting is identical to every other wrapper:
    N initial evaluations plus N per iteration.
    """
    # ---- quantum-inspired chaotic initialisation
    z = rng.uniform(0.1, 0.9)
    theta = np.empty((N, nc))
    for i in range(N):
        for j in range(nc):
            z = 4.0 * z * (1.0 - z)
            theta[i, j] = 0.5 * pi * z                   # quantum rotation angle
    X = 6.0 * (np.sin(theta) ** 2) - 3.0                 # amplitude beta^2 mapped to [-3, 3]

    P = np.array([_repair(np.sin(theta[i]) ** 2 > rng.random(nc), rng, kmin)
                  for i in range(N)])
    f = np.array([ev(p) for p in P])
    g = int(np.argmin(f))
    gb, gf, Xg = P[g].copy(), f[g], X[g].copy()

    Q = np.zeros((3, 4))
    state, stall = 0, 0

    for t in range(1, T + 1):
        tau = tau_hi - (tau_hi - tau_lo) * (t - 1) / max(1, T - 1)
        eps = eps0 * (1.0 - (t - 1) / max(1, T - 1))
        act = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[state]))
        improved = False

        for i in range(N):
            if i < N // 2:                                   # phase 1: river herd
                I = rng.integers(1, 3)
                y = X[i] + rng.random(nc) * (Xg - I * X[i])
            elif rng.random() < 0.5:                         # phase 2: predator defence
                pred = rng.uniform(-3, 3, nc)
                dist = np.abs(pred - X[i]) + 1e-9
                y = np.where(f[i] > gf,
                             pred + (2 * rng.random(nc)) / dist,
                             X[i] + rng.random(nc) * (pred - X[i]))
            else:                                            # phase 3: local escape
                lo, hi = -3.0 / t, 3.0 / t
                y = X[i] + rng.random(nc) * (lo + rng.random(nc) * (hi - lo))
            y = np.clip(y, -6, 6)
            m = _repair(_apply_tf(y, tau, P[i], act, rng), rng, kmin)
            fv = ev(m)
            if fv < f[i]:
                f[i], P[i], X[i] = fv, m, y
            if fv < gf:
                gf, gb, Xg = fv, m.copy(), y.copy()
                improved = True

        # ---- Q-learning update
        r = 1.0 if improved else 0.0
        stall = 0 if improved else stall + 1
        nxt = 0 if stall == 0 else (1 if stall < 3 else 2)
        Q[state, act] += alpha * (r + gamma_q * np.max(Q[nxt]) - Q[state, act])
        state = nxt

    return gb, gf
