"""Aggregate every experiment into the tables and figures used by the revised manuscript."""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATASETS
from stats import friedman, nemenyi_cd, holm, cliffs_delta, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

KEYS = [d[0] for d in DATASETS]
REPO = {d[0]: d[2] for d in DATASETS}
plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.family": "DejaVu Sans"})


def load(prefix):
    out = {}
    for k in KEYS:
        f = os.path.join(RES, f"{prefix}_{k}.json")
        if os.path.exists(f):
            out[k] = json.load(open(f))
    return out


def panel(data, metric):
    """(datasets x methods) matrix of a metric."""
    ks = [k for k in KEYS if k in data]
    ms = list(data[ks[0]]["agg"].keys())
    M = np.array([[data[k]["agg"][m][metric] for m in ms] for k in ks])
    return ks, ms, M


def w(name, df, floatfmt="%.3f"):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    return df


# ================================================================ MAIN COMPARISON
def main_tables():
    data = load("main")
    if not data:
        return None
    ks, ms, _ = panel(data, "auc")
    rows = []
    for m in ms:
        r = {"Method": m}
        for met, lab in [("acc", "Acc"), ("f1", "F1"), ("auc", "AUC"), ("mcc", "MCC"),
                         ("gmean", "G-mean"), ("nfeat", "#Feat"), ("time", "Time(s)"),
                         ("recall20", "Recall@20%"), ("popt20", "Popt@20%"),
                         ("ea_auc", "EA-AUC"), ("evals", "Evals"), ("uniq", "UniqEvals")]:
            r[lab] = float(np.mean([data[k]["agg"][m][met] for k in ks]))
        rows.append(r)
    t2 = pd.DataFrame(rows)
    w("table2_overall", t2)

    # per-dataset DF-GWO detail
    det = []
    for k in ks:
        a = data[k]["agg"]["DF-GWO"]
        det.append(dict(Dataset=k, Repository=REPO[k], Acc=a["acc"], F1=a["f1"],
                        AUC=a["auc"], MCC=a["mcc"], Gmean=a["gmean"],
                        NFeat=a["nfeat"], Recall20=a["recall20"],
                        Popt20=a["popt20"], EAAUC=a["ea_auc"]))
    w("table3_perdataset", pd.DataFrame(det))

    # pairwise Wilcoxon + Cliff's delta vs DF-GWO
    pr = []
    for m in ms:
        if m == "DF-GWO":
            continue
        row = {"Compared method": m}
        for met, lab in [("f1", "F1"), ("auc", "AUC"), ("mcc", "MCC")]:
            a = [data[k]["agg"]["DF-GWO"][met] for k in ks]
            b = [data[k]["agg"][m][met] for k in ks]
            row[f"p ({lab})"] = wilcoxon(a, b)
        a = [data[k]["agg"]["DF-GWO"]["auc"] for k in ks]
        b = [data[k]["agg"][m]["auc"] for k in ks]
        d, lab = cliffs_delta(a, b)
        row["delta (AUC)"] = f"{d:+.3f} ({lab})"
        an = [data[k]["agg"]["DF-GWO"]["nfeat"] for k in ks]
        bn = [data[k]["agg"][m]["nfeat"] for k in ks]
        row["p (#Feat)"] = wilcoxon(an, bn)
        dn, ln = cliffs_delta(an, bn)
        row["delta (#Feat)"] = f"{dn:+.3f} ({ln})"
        pr.append(row)
    w("table4_wilcoxon", pd.DataFrame(pr))

    # Friedman + Holm on AUC, F1, MCC and #features
    fr = []
    posthoc = {}
    for met, lab, hib in [("auc", "AUC", True), ("f1", "F1", True),
                          ("mcc", "MCC", True), ("nfeat", "#Feat", False)]:
        M = np.array([[data[k]["agg"][m][met] for m in ms] for k in ks])
        avg, chi2, pchi, F, pF = friedman(M, higher_is_better=hib)
        cd = nemenyi_cd(len(ms), len(ks))
        ci = ms.index("DF-GWO")
        ph = holm(avg, ci, len(ks), len(ms))
        posthoc[lab] = dict(ranks={m: float(a) for m, a in zip(ms, avg)},
                            cd=float(cd), chi2=chi2, p_chi=pchi, F=F, p_F=pF,
                            holm=[[ms[i], z, p, ph_] for i, z, p, ph_ in ph])
        row = {"Metric": lab, "Friedman chi2": chi2, "p (chi2)": pchi,
               "Iman-Davenport F": F, "p (F)": pF, "CD (Nemenyi, 0.05)": cd}
        for m, a in zip(ms, avg):
            row[m] = a
        fr.append(row)
    w("table5_friedman", pd.DataFrame(fr))
    json.dump(posthoc, open(os.path.join(TAB, "posthoc.json"), "w"), indent=1)
    return data, ks, ms, t2, posthoc


# ================================================================ FIGURES
def figures(data, ks, ms, t2, posthoc):
    order = ms
    # Fig 2: metric bars
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    for ax, (met, lab) in zip(axes.ravel(), [("F1", "F1-score"), ("AUC", "AUC"),
                                             ("MCC", "MCC"), ("G-mean", "G-mean")]):
        v = t2.set_index("Method").loc[order, met]
        cols = ["#c0392b" if m == "DF-GWO" else "#7f8c8d" for m in order]
        ax.bar(range(len(order)), v.values, color=cols)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=75, fontsize=6)
        ax.set_title(lab, fontsize=9)
        ax.set_ylim(min(v.values) * 0.9, max(v.values) * 1.04)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_metrics.png")); plt.close()

    # Fig 3: feature counts + parsimony/AUC scatter
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    v = t2.set_index("Method").loc[order, "#Feat"]
    a1.bar(range(len(order)), v.values,
           color=["#c0392b" if m == "DF-GWO" else "#7f8c8d" for m in order])
    a1.set_xticks(range(len(order))); a1.set_xticklabels(order, rotation=75, fontsize=6)
    a1.set_ylabel("Mean selected features"); a1.set_title("Feature-set size", fontsize=9)
    ti = t2.set_index("Method")
    pts = [(m, ti.loc[m, "#Feat"], ti.loc[m, "AUC"]) for m in order]
    front = sorted([p for p in pts
                    if not any((f2 <= p[1] and u2 >= p[2]) and (f2 < p[1] or u2 > p[2])
                               for _, f2, u2 in pts)], key=lambda p: p[1])
    a2.step([p[1] for p in front], [p[2] for p in front], where="post", lw=1.0,
            ls="--", color="#c0392b", alpha=0.6, zorder=1,
            label="non-dominated front")
    for m, x, y in pts:
        on = any(m == f[0] for f in front)
        a2.scatter(x, y, s=64 if m == "DF-GWO" else 34,
                   c="#c0392b" if m == "DF-GWO" else ("#e67e22" if on else "#34495e"),
                   marker="*" if m == "DF-GWO" else "o",
                   zorder=3 if m == "DF-GWO" else 2)
        a2.annotate(m, (x, y), fontsize=5.5, xytext=(3.5, 3), textcoords="offset points",
                    color="#c0392b" if m == "DF-GWO" else "#2c3e50")
    a2.legend(fontsize=6, loc="lower right")
    a2.set_xlabel("Mean number of selected features"); a2.set_ylabel("Mean AUC")
    a2.set_title("Parsimony-accuracy plane", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_parsimony.png")); plt.close()

    # Fig 4: per-dataset AUC for a readable subset of methods
    sel = [m for m in ["DF-GWO", "GWO", "EMWS", "OBCGWO", "BHO", "AllFeatures"] if m in ms]
    plt.figure(figsize=(9.5, 3.4))
    mk = dict(zip(sel, ["o", "^", "s", "v", "D", "x"]))
    for m in sel:
        plt.plot(range(len(ks)), [data[k]["agg"][m]["auc"] for k in ks],
                 marker=mk[m], ms=3.6, lw=1.1, label=m)
    plt.xticks(range(len(ks)), ks, rotation=75, fontsize=6)
    plt.ylabel("AUC"); plt.legend(fontsize=6, ncol=6); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_perdataset_auc.png")); plt.close()

    # Fig 5: effort-aware per dataset
    plt.figure(figsize=(9.5, 3.4))
    for met, lab, mk_ in [("recall20", "Recall@20%LOC", "o"),
                          ("popt20", "Popt@20%", "s"), ("ea_auc", "EA-AUC", "^")]:
        v = [data[k]["agg"]["DF-GWO"][met] for k in ks]
        plt.plot(range(len(ks)), v, marker=mk_, ms=3.6, lw=1.1, label=lab)
        plt.axhline(np.mean(v), ls="--", lw=0.7, alpha=0.5)
    plt.xticks(range(len(ks)), ks, rotation=75, fontsize=6)
    plt.ylabel("Effort-aware score"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_effort.png")); plt.close()

    # Fig 6: Nemenyi critical-difference diagram (AUC and #Feat)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6))
    for ax, lab in zip(axes, ["AUC", "#Feat"]):
        ranks = posthoc[lab]["ranks"]; cd = posthoc[lab]["cd"]
        items = sorted(ranks.items(), key=lambda kv: kv[1])
        names = [i[0] for i in items]; vals = [i[1] for i in items]
        ax.plot(vals, [0] * len(vals), "o", color="#34495e", ms=4.5, zorder=3)
        span = max(vals) - min(vals)
        lvl = [0.0] * len(vals)          # stack labels that would collide
        for i in range(len(vals)):
            k = 0
            while any(abs(vals[i] - vals[j]) < 0.055 * span and lvl[j] == k
                      for j in range(i)):
                k += 1
            lvl[i] = k
        for i, (nm, v) in enumerate(items):
            up = (i % 2 == 0)
            off = (18 + 13 * lvl[i]) * (1 if up else -1)
            ax.annotate(nm, (v, 0), fontsize=6.2, ha="center",
                        va="bottom" if up else "top",
                        xytext=(0, off), textcoords="offset points",
                        color="#c0392b" if nm == "DF-GWO" else "#2c3e50",
                        weight="bold" if nm == "DF-GWO" else "normal",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#95a5a6",
                                        shrinkA=0, shrinkB=2))
        ax.plot([vals[0], vals[0] + cd], [-0.06, -0.06], lw=2.5, color="#c0392b",
                solid_capstyle="butt", zorder=4)
        ax.annotate(f"CD = {cd:.2f}", (vals[0] + cd / 2, -0.06), fontsize=7,
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    color="#c0392b")
        ax.set_yticks([]); ax.set_ylim(-1.0, 1.0); ax.grid(False)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel(f"Average Friedman rank ({lab}; lower is better)", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig6_cd.png")); plt.close()

    # Fig 7: convergence of the composite fitness
    plt.figure(figsize=(5.2, 3.4))
    for m in [x for x in ["DF-GWO", "GWO", "OBCGWO", "SA-bGWO", "BHO", "EMWS", "PSO", "GA"]
              if x in ms]:
        cs = [data[k]["curves"][m] for k in ks if m in data[k]["curves"]]
        if cs:
            plt.plot(np.mean(cs, axis=0), lw=1.2, label=m)
    plt.xlabel("Iteration"); plt.ylabel("Best composite fitness (lower is better)")
    plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_convergence.png")); plt.close()


# ================================================================ ABLATION
def ablation():
    data = load("abl")
    if not data:
        return None
    ks = [k for k in KEYS if k in data]
    ms = list(data[ks[0]]["agg"].keys())
    rows = []
    for m in ms:
        r = {"Configuration": m}
        for met, lab in [("f1", "F1"), ("auc", "AUC"), ("mcc", "MCC"),
                         ("gmean", "G-mean"), ("nfeat", "#Feat"),
                         ("auc_std", "AUC SD"), ("time", "Time(s)")]:
            r[lab] = float(np.mean([data[k]["agg"][m][met] for k in ks]))
        a = [data[k]["agg"]["DF-GWO"]["auc"] for k in ks]
        b = [data[k]["agg"][m]["auc"] for k in ks]
        r["p vs full (AUC)"] = wilcoxon(a, b) if m != "DF-GWO" else np.nan
        an = [data[k]["agg"]["DF-GWO"]["nfeat"] for k in ks]
        bn = [data[k]["agg"][m]["nfeat"] for k in ks]
        r["p vs full (#Feat)"] = wilcoxon(an, bn) if m != "DF-GWO" else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    w("table6_ablation", df)

    M = np.array([[data[k]["agg"][m]["auc"] for m in ms] for k in ks])
    avg, chi2, pchi, F, pF = friedman(M)
    json.dump(dict(ranks={m: float(a) for m, a in zip(ms, avg)}, chi2=chi2, p_chi=pchi,
                   F=F, p_F=pF, cd=nemenyi_cd(len(ms), len(ks))),
              open(os.path.join(TAB, "ablation_friedman.json"), "w"), indent=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    short = {"DF-GWO": "complete", "DF-GWO base": "base", "DF-GWO w/o group": "-group",
             "DF-GWO w/o chaos": "-chaos", "DF-GWO w/o Levy": "-Levy",
             "DF-GWO w/o OBL": "-OBL", "DF-GWO w/o restart": "-restart",
             "DF-GWO w/o LS": "-local search", "DF-GWO chaos only": "+chaos",
             "DF-GWO Levy only": "+Levy", "DF-GWO OBL only": "+OBL",
             "DF-GWO restart only": "+restart", "DF-GWO LS only": "+local search"}
    for _, r in df.iterrows():
        m = r["Configuration"]
        full_cfg = m == "DF-GWO"
        base_cfg = m == "DF-GWO base"
        col = "#c0392b" if full_cfg else ("#2980b9" if base_cfg else
                                          ("#7f8c8d" if m.startswith("DF-GWO w/o")
                                           else "#27ae60"))
        a1.scatter(r["#Feat"], r["AUC"], s=90 if full_cfg else 34, color=col,
                   marker="*" if full_cfg else ("s" if base_cfg else "o"),
                   zorder=3 if full_cfg else 2)
        a1.annotate(short.get(m, m), (r["#Feat"], r["AUC"]), fontsize=6,
                    xytext=(4, 3), textcoords="offset points", color=col)
    a1.set_xlabel("Mean number of selected features"); a1.set_ylabel("Mean AUC")
    a1.set_title("Ablation configurations on the parsimony-accuracy plane", fontsize=9)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker="*", ls="", color="#c0392b", label="complete"),
                       Line2D([], [], marker="s", ls="", color="#2980b9", label="base"),
                       Line2D([], [], marker="o", ls="", color="#7f8c8d", label="one removed"),
                       Line2D([], [], marker="o", ls="", color="#27ae60", label="one added")],
              fontsize=6, loc="lower right")
    for m in [x for x in ms if x in ("DF-GWO", "DF-GWO w/o chaos", "DF-GWO w/o Levy",
                                     "DF-GWO w/o OBL", "DF-GWO w/o restart",
                                     "DF-GWO w/o LS", "DF-GWO base")]:
        cs = [data[k]["curves"][m] for k in ks if m in data[k]["curves"]]
        if cs:
            a2.plot(np.mean(cs, axis=0), lw=1.2, label=short.get(m, m))
    a2.set_xlabel("Iteration"); a2.set_ylabel("Best composite fitness (minimised)")
    a2.set_title("Mean convergence", fontsize=9)
    a2.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig8_ablation.png")); plt.close()
    return df


# ================================================================ SENSITIVITY
FACTOR_TITLE = {"beta": "gamma  (size penalty)",
                "chaos_lo": "lambda  (chaotic floor)",
                "tau": "tau  (correlation threshold)",
                "weights": "w  (fitness weights)"}


def sensitivity():
    rows = []
    for f in glob.glob(os.path.join(RES, "sens_*.json")):
        j = json.load(open(f))
        rows.append(dict(family=j["family"], value=str(j["value"]), dataset=j["dataset"],
                         **{k: j["agg"][k] for k in ("f1", "auc", "mcc", "gmean",
                                                     "nfeat", "auc_std")}))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    g = (df.groupby(["family", "value"])
           .agg(F1=("f1", "mean"), AUC=("auc", "mean"), MCC=("mcc", "mean"),
                Gmean=("gmean", "mean"), NFeat=("nfeat", "mean"),
                AUC_SD=("auc_std", "mean"), n=("dataset", "count"))
           .reset_index())
    w("table7_sensitivity", g)

    # Friedman across levels within each family
    fr = []
    for fam in g["family"].unique():
        sub = df[df.family == fam]
        lv = sorted(sub.value.unique())
        piv = sub.pivot_table(index="dataset", columns="value", values="auc")
        piv = piv[lv].dropna()
        if piv.shape[0] >= 5 and piv.shape[1] >= 2:
            avg, chi2, pchi, F, pF = friedman(piv.values)
            fr.append(dict(Family=fam, Levels=len(lv), chi2=chi2, p_chi2=pchi,
                           F=F, p_F=pF,
                           ranks="; ".join(f"{a}:{b:.2f}" for a, b in zip(lv, avg))))
    w("table8_sensitivity_friedman", pd.DataFrame(fr))

    fams = list(g["family"].unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(3.1 * len(fams), 3.0))
    if len(fams) == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams):
        s = g[g.family == fam].sort_values("value")
        x = range(len(s))
        ax.plot(x, s["AUC"], "o-", color="#c0392b", label="AUC")
        ax.plot(x, s["F1"], "s-", color="#2980b9", label="F1")
        ax.set_xticks(list(x)); ax.set_xticklabels(s["value"], rotation=45, fontsize=6)
        ax2 = ax.twinx(); ax2.bar(x, s["NFeat"], alpha=0.18, color="#7f8c8d")
        ax2.set_ylabel("#features", fontsize=7); ax2.grid(False)
        # the paper writes the size penalty as gamma (beta denotes the beta wolf)
        ax.set_title(FACTOR_TITLE.get(fam, fam), fontsize=8.5); ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig9_sensitivity.png")); plt.close()
    return g


# ================================================================ BENCHMARKS
def benchmarks():
    fb = os.path.join(RES, "bench_functions.json")
    if not os.path.exists(fb):
        return None
    J = json.load(open(fb))
    ms = list(next(iter(J.values()))["res"].keys())
    rows = []
    for fn, blk in J.items():
        for m in ms:
            r = blk["res"][m]
            rows.append(dict(Function=fn, Method=m, Mean=r["mean"], SD=r["std"],
                             Best=r["best"]))
    df = pd.DataFrame(rows)
    piv = df.pivot(index="Function", columns="Method", values="Mean")
    sd = df.pivot(index="Function", columns="Method", values="SD")
    fo = [f"F{i}" for i in range(1, 14)]
    piv = piv.loc[fo, ms]; sd = sd.loc[fo, ms]
    tab = piv.copy()
    for m in ms:
        tab[m] = [f"{v:.3e}" + "\n" + f"({s:.1e})" for v, s in zip(piv[m], sd[m])]
    tab = tab.reset_index()
    w("table9_benchmark_functions", tab)

    avg, chi2, pchi, F, pF = friedman(piv.values, higher_is_better=False)
    json.dump(dict(ranks={m: float(a) for m, a in zip(ms, avg)}, chi2=chi2, p_chi=pchi,
                   F=F, p_F=pF, cd=nemenyi_cd(len(ms), len(fo))),
              open(os.path.join(TAB, "bench_friedman.json"), "w"), indent=1)

    wl = []
    for m in ms:
        if m == "DF-GWO":
            continue
        a = [J[f]["res"]["DF-GWO"]["mean"] for f in fo]
        b = [J[f]["res"][m]["mean"] for f in fo]
        tie = sum(abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))
        win = sum(x < y and abs(x - y) > 1e-12 * max(1.0, abs(x), abs(y))
                  for x, y in zip(a, b))
        wl.append(dict(Method=m, Win=win, Tie=tie, Loss=len(fo) - win - tie,
                       p_Wilcoxon=wilcoxon(a, b)))
    w("table10_benchmark_wilcoxon", pd.DataFrame(wl))

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for ax, fn in zip(axes.ravel(), ["F1", "F5", "F9", "F10", "F11", "F13"]):
        FLOOR = 1e-20                      # values below this are clipped for display
        for m in ms:
            c = np.array(J[fn]["curves"][m], float)
            shift = 0.0
            if c.min() < 0:                # F8 is negative: plot the gap to the best value
                shift = -min(np.min([np.min(J[fn]["curves"][x]) for x in ms]), 0.0)
            ax.semilogy(np.arange(len(c)) * 10, np.maximum(c + shift, FLOOR),
                        lw=1.1, label=m)
        vals = np.concatenate([np.maximum(np.array(J[fn]["curves"][x], float), FLOOR)
                               for x in ms])
        ax.set_ylim(bottom=max(FLOOR / 5, float(vals.min()) * 0.3))
        ax.set_title(fn, fontsize=9); ax.set_xlabel("Iteration", fontsize=7)
        ax.set_ylabel("Best fitness" + (" (shifted)" if fn == "F8" else ""), fontsize=7)
    axes.ravel()[0].legend(fontsize=5.5, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig10_bench_convergence.png"))
    plt.close()

    fe = os.path.join(RES, "bench_engineering.json")
    if os.path.exists(fe):
        E = json.load(open(fe))
        rows = []
        for pn, blk in E.items():
            for m in ms:
                r = blk["res"][m]
                rows.append(dict(Problem=pn, Method=m, Best=r["best"], Mean=r["mean"],
                                 Worst=r["worst"], SD=r["std"],
                                 Reference=blk["known"]))
        w("table11_engineering", pd.DataFrame(rows))
    return tab


if __name__ == "__main__":
    r = main_tables()
    if r:
        data, ks, ms, t2, ph = r
        figures(data, ks, ms, t2, ph)
        print(t2.to_string(index=False))
    a = ablation()
    if a is not None:
        print("\nABLATION\n", a.to_string(index=False))
    s = sensitivity()
    if s is not None:
        print("\nSENSITIVITY\n", s.to_string(index=False))
    b = benchmarks()
    if b is not None:
        print("\nBENCHMARK\n", b.to_string(index=False))
    print("\nwrote tables ->", TAB, "\nfigures ->", FIG)
