"""Figure 1: revised DF-GWO pipeline diagram."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(figsize=(11.2, 5.0))
ax.set_xlim(0, 114); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, text, fc, fs=8.2, bold=False, ec="#37474f"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2, style="-|>", ls="-", col="#37474f", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 lw=1.2, color=col, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# --- data / fold layer
box(1.5, 39, 20, 8.5, "Twenty projects\nPROMISE / AEEEM / ReLink\n(n = 56-1862, d = 20-61)", "#d6eaf8")
box(24.5, 39, 19, 8.5, "Stratified 5-fold CV\n(outer folds fixed and\nshared by all methods)", "#d6eaf8")
arrow(21.5, 43.2, 24.5, 43.2)

box(46.5, 39, 22, 8.5, "Training partition only:\nmedian impute, drop constant,\nmin-max scale, SMOTE (k = 5)", "#d5f5e3")
arrow(43.5, 43.2, 46.5, 43.2)
box(92, 39, 19, 8.5, "Held-out test fold\n(untouched until the\nfinal Random Forest)", "#fdebd0")
arrow(34, 47.5, 97, 47.5, ls=(0, (4, 3)), col="#b9770e", rad=-0.11)

# --- stage 1
box(3, 27, 26, 8.5, "STAGE 1  Dynamic feature grouping\nPearson |rho| > tau = 0.85, union-find merge;\n"
                    "MI representative + high-MI singletons\n-> candidate pool C", "#fdebd0", bold=False)
arrow(50, 39, 16, 35.5, rad=0.12)

# --- stage 2 / 3 loop
box(33, 20.5, 32, 15, "", "#fadbd8", ec="#c0392b")
ax.text(49, 34.0, "Main loop:  t = 1 ... T = 30", ha="center", fontsize=8.6, weight="bold",
        color="#922b21")
box(34.5, 27.5, 29, 5.6, "STAGE 2  Chaos-driven search\ncosine a(t) Eq.(6) | logistic C' Eq.(7) | Levy (t > T/2)",
    "#ffffff", fs=7.9)
box(34.5, 21.5, 29, 5.4, "STAGE 3  Composite fitness Eq.(8)\n0.4 F1 + 0.4 AUC + 0.2 MCC* + beta |S|/|C|\n"
                         "on internal 20% validation slice (L1-logistic probe)", "#ffffff", fs=7.4)
arrow(29, 30.5, 34.5, 30.5)
arrow(49, 27.5, 49, 27.0)

# --- stage 4
box(69, 20.5, 30, 15, "", "#e8daef", ec="#6c3483")
ax.text(84, 34.0, "STAGE 4  Population management", ha="center", fontsize=8.6, weight="bold",
        color="#5b2c6f")
box(70.5, 29.5, 27, 3.6, "Opposition-based initialisation (t = 0)", "#ffffff", fs=7.9)
box(70.5, 25.4, 27, 3.6, "Adaptive restart after 5 stalled iterations", "#ffffff", fs=7.9)
box(70.5, 21.3, 27, 3.6, "Bounded local search (2 passes, after loop)", "#ffffff", fs=7.9)
arrow(65, 28, 69, 28)
arrow(69, 23.5, 65, 23.5)

# --- output layer
box(33, 10.5, 32, 6.5, "Selected subset S*  (|S*| >= 3)", "#d5f5e3", bold=True)
arrow(49, 20.5, 49, 17)
box(3, 10.5, 26, 6.5, "Random Forest scorer\n100 trees, balanced class weights", "#d6eaf8")
arrow(33, 13.7, 29, 13.7)
box(69, 10.5, 30, 6.5, "Predicted defect probabilities\non the held-out test fold", "#fdebd0")
arrow(65, 13.7, 69, 13.7)
arrow(105, 39, 105, 13.7, style="-")
arrow(105, 13.7, 99, 13.7)

# --- evaluation layer
box(3, 1.5, 45, 6.5, "Predictive metrics\nAccuracy | F1 | AUC | MCC | G-mean", "#eaeded")
box(51, 1.5, 48, 6.5, "Effort-aware metrics (defect density ranking)\nRecall@20%LOC | Popt@20% | EA-AUC",
    "#eaeded")
arrow(16, 10.5, 16, 8)
arrow(84, 10.5, 84, 8)

ax.text(65, 49.6, "test partition of each fold", ha="center", fontsize=7.2, color="#b9770e", style="italic")
ax.set_ylim(0, 52)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_pipeline.png"), dpi=300, bbox_inches="tight")
print("wrote", os.path.join(FIG, "fig1_pipeline.png"))
