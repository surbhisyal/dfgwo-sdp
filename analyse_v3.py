"""Tables and figures for the revision-3 continuous experiments.

Builds, from results/cec2022_D*.json, results/bench_functions_v3.json and
results/bench_engineering_v3.json:

    table12_cec_D10.csv          mean (std) solution error, 12 functions x 12 algorithms
    table13_cec_D20.csv          the same at D = 20
    table14_cec_friedman.csv     Friedman / Iman-Davenport / Nemenyi CD per dimension
    table15_cec_posthoc.csv      Holm post-hoc and Wilcoxon W/T/L against DF-GWO
    table16_cec_byclass.csv      mean Friedman rank per function class
    table17_gwo_family.csv       the GWO family alone, by class
    table18_classic.csv          F1-F13 at D = 30, expanded algorithm set
    table19_engineering.csv      engineering problems with feasibility rates

    fig11_cec_convergence.png    median convergence by function class
    fig12_cec_cd.png             critical-difference diagram over the 24 CEC problems
    fig13_cec_box.png            error distributions by function class
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats import friedman, nemenyi_cd, holm, wilcoxon, cliffs_delta

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG, TAB = (os.path.join(HERE, d) for d in ("results", "figures", "tables"))
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.family": "DejaVu Sans"})

CLASSES = ["Unimodal", "Basic", "Hybrid", "Composition"]
# the reviewer asks for unimodal / multimodal / hybrid / composition; CEC-2022 calls its
# four basic multimodal functions simply "basic", so the panels spell that out
DISPLAY = {"Unimodal": "Unimodal", "Basic": "Basic multimodal",
           "Hybrid": "Hybrid", "Composition": "Composition"}
CTRL = "DF-GWO"


def _w(name, df):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    print("wrote", name, df.shape)
    return df


def _load(fn):
    p = os.path.join(RES, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def _fmt(m, s):
    return f"{m:.3e} ({s:.2e})"


# ------------------------------------------------------------------ CEC-2022
def cec_table(blob, dim):
    algs = blob["_meta"]["algs"]
    funcs = [k for k in blob if not k.startswith("_")]
    rows = []
    for fn in funcs:
        d = blob[fn]
        best = min(d["res"][a]["mean"] for a in algs)
        r = {"Function": fn, "Class": d["cls"]}
        for a in algs:
            v = d["res"][a]
            star = "*" if v["mean"] <= best * (1 + 1e-12) else ""
            r[a] = _fmt(v["mean"], v["std"]) + star
        rows.append(r)
    M = np.array([[blob[fn]["res"][a]["mean"] for a in algs] for fn in funcs])
    ranks, chi2, p_chi, F, p_F = friedman(M, higher_is_better=False)
    rows.append({"Function": "Mean Friedman rank", "Class": "",
                 **{a: f"{ranks[i]:.2f}" for i, a in enumerate(algs)}})
    return _w(f"table{12 if dim == 10 else 13}_cec_D{dim}", pd.DataFrame(rows)), \
        (funcs, algs, M, ranks, chi2, p_chi, F, p_F)


def cec_analysis(parts10, parts20, blobs):
    """Friedman per dimension and pooled over all 24 problems, plus post-hoc."""
    fr = []
    pooled_rows, algs = [], parts10[1]
    for dim, parts in ((10, parts10), (20, parts20)):
        funcs, algs, M, ranks, chi2, p_chi, F, p_F = parts
        n, k = M.shape
        cd = nemenyi_cd(k, n)
        fr.append(dict(Suite=f"CEC-2022 D={dim}", n=n, k=k,
                       Chi2=round(chi2, 2), p_chi=f"{p_chi:.3e}",
                       ImanDavenport_F=round(F, 2), p_F=f"{p_F:.3e}",
                       Nemenyi_CD=round(cd, 3),
                       Best=algs[int(np.argmin(ranks))],
                       DFGWO_rank=round(float(ranks[algs.index(CTRL)]), 2)))
        pooled_rows.append(M)
    Mall = np.vstack(pooled_rows)
    ranks, chi2, p_chi, F, p_F = friedman(Mall, higher_is_better=False)
    n, k = Mall.shape
    cd = nemenyi_cd(k, n)
    fr.append(dict(Suite="CEC-2022 pooled (D=10 and D=20)", n=n, k=k,
                   Chi2=round(chi2, 2), p_chi=f"{p_chi:.3e}",
                   ImanDavenport_F=round(F, 2), p_F=f"{p_F:.3e}",
                   Nemenyi_CD=round(cd, 3), Best=algs[int(np.argmin(ranks))],
                   DFGWO_rank=round(float(ranks[algs.index(CTRL)]), 2)))
    _w("table14_cec_friedman", pd.DataFrame(fr))

    # Holm post-hoc against DF-GWO + paired Wilcoxon over the 24 problems
    ci = algs.index(CTRL)
    hol = {i: (z, pr, ph) for i, z, pr, ph in holm(ranks, ci, n, k)}
    rows = []
    for i, a in enumerate(algs):
        if a == CTRL:
            continue
        x, y = Mall[:, ci], Mall[:, i]
        win = int(np.sum(x < y))
        loss = int(np.sum(x > y))
        tie = int(n - win - loss)
        dlt, lab = cliffs_delta(x, y)
        z, pr, ph = hol[i]
        rows.append(dict(Comparator=a, Rank=round(float(ranks[i]), 2),
                         DFGWO_better=win, Tie=tie, DFGWO_worse=loss,
                         Wilcoxon_p=f"{wilcoxon(x, y):.3e}",
                         Holm_z=round(z, 2), Holm_p=f"{ph:.3e}",
                         Significant="yes" if ph < 0.05 else "no",
                         Cliffs_delta=round(dlt, 3), Magnitude=lab))
    rows.sort(key=lambda r: r["Rank"])
    _w("table15_cec_posthoc", pd.DataFrame(rows))

    # rank per function class, pooled over both dimensions
    byc = []
    for cls in CLASSES:
        sub = []
        for dim, blob in blobs.items():
            fs = [f for f in blob if not f.startswith("_") and blob[f]["cls"] == cls]
            sub += [[blob[f]["res"][a]["mean"] for a in algs] for f in fs]
        Mc = np.asarray(sub, float)
        rk, *_ = friedman(Mc, higher_is_better=False)
        byc.append({"Class": cls, "n_problems": Mc.shape[0],
                    **{a: round(float(rk[i]), 2) for i, a in enumerate(algs)}})
    allrk, *_ = friedman(Mall, higher_is_better=False)
    byc.append({"Class": "All", "n_problems": Mall.shape[0],
                **{a: round(float(allrk[i]), 2) for i, a in enumerate(algs)}})
    _w("table16_cec_byclass", pd.DataFrame(byc))

    # the GWO family on its own -- reviewer 1, comment 2
    from cbench import GWO_FAMILY
    fam = [a for a in GWO_FAMILY if a in algs]
    idx = [algs.index(a) for a in fam]
    rows = []
    for cls in CLASSES + ["All"]:
        sub = []
        for dim, blob in blobs.items():
            fs = [f for f in blob if not f.startswith("_")
                  and (cls == "All" or blob[f]["cls"] == cls)]
            sub += [[blob[f]["res"][a]["mean"] for a in fam] for f in fs]
        Mc = np.asarray(sub, float)
        rk, chi2, p_chi, F, p_F = friedman(Mc, higher_is_better=False)
        rows.append({"Class": cls, "n_problems": Mc.shape[0],
                     **{a: round(float(rk[i]), 2) for i, a in enumerate(fam)},
                     "p_Friedman": f"{p_F:.3e}"})
    _w("table17_gwo_family", pd.DataFrame(rows))
    return ranks, algs, Mall, cd


# ------------------------------------------------------------------ figures
def cec_figures(blobs, ranks, algs, cd):
    blob = blobs[20]
    show = ["DF-GWO", "GWO", "I-GWO", "RW-GWO", "mGWO", "CGWO", "HO", "SHO", "PSO", "WOA"]
    show = [a for a in show if a in algs]
    colors = plt.cm.tab10(np.linspace(0, 1, len(show)))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, cls in zip(axes.ravel(), CLASSES):
        fs = [f for f in blob if not f.startswith("_") and blob[f]["cls"] == cls]
        max_fe = blob["_meta"]["max_fe"]
        for c, a in zip(colors, show):
            # mean over the functions of this class of the per-function median curve
            C = np.array([blob[f]["curves"][a]["median"] for f in fs], float)
            y = np.maximum(C.mean(axis=0), 1e-8)
            x = np.linspace(max_fe / len(y), max_fe, len(y))
            ax.semilogy(x, y, lw=1.3, color=c,
                        label=a, zorder=5 if a == CTRL else 2)
        ax.set_title(f"{DISPLAY[cls]}  ({len(fs)} function{'s' if len(fs) > 1 else ''}, D=20)",
                     fontsize=9)
        ax.set_xlabel("Function evaluations", fontsize=8)
        ax.set_ylabel("Solution error  F(x) - F*", fontsize=8)
    axes[0, 0].legend(fontsize=6.4, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig11_cec_convergence.png"))
    plt.close()

    # critical-difference diagram over the pooled 24 problems
    fig, ax = plt.subplots(figsize=(9, 3.0))
    items = sorted(zip(algs, ranks), key=lambda kv: kv[1])
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    ax.plot(vals, [0] * len(vals), "o", color="#34495e", ms=4.5, zorder=3)
    span = max(vals) - min(vals)
    lvl = [0] * len(vals)
    for i in range(len(vals)):
        k = 0
        while any(abs(vals[i] - vals[j]) < 0.055 * span and lvl[j] == k
                  for j in range(i)):
            k += 1
        lvl[i] = k
    for i, (nm, v) in enumerate(items):
        up = (i % 2 == 0)
        off = (18 + 13 * lvl[i]) * (1 if up else -1)
        ax.annotate(nm, (v, 0), fontsize=6.6, ha="center",
                    va="bottom" if up else "top",
                    xytext=(0, off), textcoords="offset points",
                    color="#c0392b" if nm == CTRL else "#2c3e50",
                    weight="bold" if nm == CTRL else "normal",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#95a5a6",
                                    shrinkA=0, shrinkB=2))
    ax.plot([vals[0], vals[0] + cd], [-0.06, -0.06], lw=2.5, color="#c0392b",
            solid_capstyle="butt", zorder=4)
    ax.annotate(f"CD = {cd:.2f}", (vals[0] + cd / 2, -0.06), fontsize=7,
                xytext=(0, 4), textcoords="offset points", ha="center", color="#c0392b")
    ax.set_yticks([])
    ax.set_ylim(-1.0, 1.0)
    ax.grid(False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Average Friedman rank over the 24 CEC-2022 problems "
                  "(D = 10 and D = 20; lower is better)", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig12_cec_cd.png"))
    plt.close()

    # error distributions by class
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2), sharey=False)
    for ax, cls in zip(axes, CLASSES):
        data = []
        for a in show:
            v = []
            for dim, bl in blobs.items():
                for f in [x for x in bl if not x.startswith("_") and bl[x]["cls"] == cls]:
                    v += bl[f]["res"][a]["vals"]
            data.append(np.log10(np.maximum(v, 1e-8)))
        bp = ax.boxplot(data, labels=show, showfliers=False, patch_artist=True,
                        medianprops=dict(color="#2c3e50"))
        for patch, a in zip(bp["boxes"], show):
            patch.set_facecolor("#e74c3c" if a == CTRL else "#bdc3c7")
            patch.set_alpha(0.75)
        ax.set_title(DISPLAY[cls], fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=6.2)
        ax.set_ylabel("log10 solution error", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig13_cec_box.png"))
    plt.close()


# ------------------------------------------------------------------ F1-F13
def classic_table(blob):
    algs = blob["_meta"]["algs"]
    funcs = [k for k in blob if not k.startswith("_")]
    rows = []
    for fn in funcs:
        d = blob[fn]
        r = {"Function": fn, "Class": d["cls"], "Optimum": d["opt"]}
        for a in algs:
            v = d["res"][a]
            r[a] = _fmt(v["mean"], v["std"])
        rows.append(r)
    M = np.array([[blob[fn]["res"][a]["mean"] for a in algs] for fn in funcs])
    ranks, chi2, p_chi, F, p_F = friedman(M, higher_is_better=False)
    rows.append({"Function": "Mean Friedman rank", "Class": "", "Optimum": "",
                 **{a: f"{ranks[i]:.2f}" for i, a in enumerate(algs)}})
    _w("table18_classic", pd.DataFrame(rows))

    # convergence on six representative classical functions, 12 algorithms
    show = [a for a in ["DF-GWO", "GWO", "I-GWO", "RW-GWO", "mGWO", "CGWO", "HO", "SHO",
                        "PSO", "WOA"] if a in algs]
    colors = plt.cm.tab10(np.linspace(0, 1, len(show)))
    pick = [f for f in ["F1", "F5", "F7", "F9", "F10", "F12"] if f in funcs]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0))
    max_fe = blob["_meta"]["max_fe"]
    for ax, fn in zip(axes.ravel(), pick):
        for c, a in zip(colors, show):
            y = np.asarray(blob[fn]["curves"][a]["median"], float)
            off = blob[fn]["opt"]
            y = np.maximum(y - off, 1e-20)          # shift so the optimum sits at zero
            x = np.linspace(max_fe / len(y), max_fe, len(y))
            ax.semilogy(x, y, lw=1.2, color=c, label=a, zorder=5 if a == CTRL else 2)
        ax.set_title(f"{fn}  ({blob[fn]['cls']}, D=30)", fontsize=9)
        ax.set_xlabel("Function evaluations", fontsize=8)
        ax.set_ylabel("Best value - optimum", fontsize=8)
    axes[0, 0].legend(fontsize=6.0, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig14_classic_convergence.png"))
    plt.close()
    return ranks, algs, M


# ------------------------------------------------------------------ engineering
def eng_table(blob):
    algs = blob["_meta"]["algs"]
    probs = [k for k in blob if not k.startswith("_")]
    rows = []
    for p in probs:
        d = blob[p]
        feas_algs = [a for a in algs if d["res"][a]["n_feasible"] > 0]
        best_val = min(d["res"][a]["best"] for a in feas_algs) if feas_algs else np.nan
        for a in algs:
            v = d["res"][a]
            rows.append(dict(Problem=p, Method=a,
                             Best=v["best"], Mean=v["mean"], Worst=v["worst"],
                             SD=v["std"],
                             Feasible=f'{v["n_feasible"]}/{blob["_meta"]["nrun"]}',
                             MaxViolation=f'{v["max_viol"]:.2e}',
                             IsBest="yes" if np.isfinite(v["best"])
                                    and v["best"] <= best_val * (1 + 1e-9) else "",
                             Reference=d["ref"]))
    _w("table19_engineering", pd.DataFrame(rows))

    # rank over the four problems on the mean feasible objective
    M = np.array([[blob[p]["res"][a]["mean"] if np.isfinite(blob[p]["res"][a]["mean"])
                   else 1e30 for a in algs] for p in probs])
    ranks, chi2, p_chi, F, p_F = friedman(M, higher_is_better=False)
    return ranks, algs, p_F


if __name__ == "__main__":
    b10, b20 = _load("cec2022_D10.json"), _load("cec2022_D20.json")
    if b10 and b20:
        _, p10 = cec_table(b10, 10)
        _, p20 = cec_table(b20, 20)
        blobs = {10: b10, 20: b20}
        ranks, algs, Mall, cd = cec_analysis(p10, p20, blobs)
        cec_figures(blobs, ranks, algs, cd)
        print("\nCEC-2022 pooled ranks (lower is better):")
        for a, r in sorted(zip(algs, ranks), key=lambda kv: kv[1]):
            print(f"   {a:8s} {r:5.2f}")

    bc = _load("bench_functions_v3.json")
    if bc:
        r, a, _ = classic_table(bc)
        print("\nF1-F13 ranks:", {x: round(float(y), 2) for x, y in
                                  sorted(zip(a, r), key=lambda kv: kv[1])})

    be = _load("bench_engineering_v3.json")
    if be:
        r, a, pF = eng_table(be)
        print("\nEngineering ranks:", {x: round(float(y), 2) for x, y in
                                       sorted(zip(a, r), key=lambda kv: kv[1])},
              f"(Friedman p = {pF:.3e})")
    print("\nDONE analyse_v3")
