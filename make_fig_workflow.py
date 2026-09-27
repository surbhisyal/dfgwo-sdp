"""Figure 2: DF-GWO algorithmic workflow (Reviewer 1, comment 5).

Figure 1 shows the experimental pipeline -- where the data comes from, what is fitted on
which partition, and how the model is scored. This figure shows something different: the
control flow of the algorithm itself, including the branch points that Algorithm 1
encodes as conditionals, so that the pseudocode and the diagram can be read against each
other line by line.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

W, H = 100.0, 143.0
fig, ax = plt.subplots(figsize=(8.0, 11.4))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

C_STAGE1 = "#fdebd0"
C_INIT = "#d6eaf8"
C_PROC = "#ffffff"
C_DEC = "#fcf3cf"
C_OUT = "#d5f5e3"
C_SIDE = "#f5eef8"
EC = "#37474f"


def box(cx, cy, w, h, text, fc=C_PROC, fs=7.6, bold=False, ec=EC, style="round"):
    bs = "round,pad=0.5,rounding_size=1.6" if style == "round" else "square,pad=0.5"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=bs,
                                fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            weight="bold" if bold else "normal", linespacing=1.4)


def diamond(cx, cy, w, h, text, fs=7.2):
    ax.add_patch(Polygon([[cx, cy + h / 2], [cx + w / 2, cy],
                          [cx, cy - h / 2], [cx - w / 2, cy]],
                         closed=True, fc=C_DEC, ec=EC, lw=1.0, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            linespacing=1.3)


def arrow(p1, p2, style="-|>", ls="-", col=EC, rad=0.0, lw=1.05, z=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=10,
                                 lw=lw, color=col, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def elbow(pts, col=EC, ls="-", lw=1.05):
    """Orthogonal polyline with an arrowhead on the final segment."""
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-", mutation_scale=10,
                                     lw=lw, color=col, linestyle=ls, zorder=2))
    arrow(pts[-2], pts[-1], col=col, ls=ls, lw=lw)


def tag(x, y, s, col="#6c3483"):
    ax.text(x, y, s, fontsize=6.6, color=col, ha="center", va="center",
            style="italic", zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.6))


CX = 44.0

# ------------------------------------------------------------------ main-loop panel
ax.add_patch(Rectangle((3, 3.0), 96, 101.5, fc="#fdf2f2", ec="#c0392b", lw=1.1,
                       linestyle=(0, (5, 3)), zorder=0))
ax.text(5.5, 102.0, "Main loop   t = 1 ... T", fontsize=8.4, weight="bold",
        color="#922b21", ha="left", zorder=5)

ax.add_patch(Rectangle((10, 31.5), 88, 56.0, fc="#ffffff", ec="#7f8c8d", lw=0.8,
                       linestyle=(0, (3, 2)), zorder=1))
ax.text(12, 85.6, "for each wolf  i = 1 ... N", fontsize=7.4, weight="bold",
        color="#566573", ha="left", zorder=5)

# ------------------------------------------------------------------ nodes
box(CX, 138.5, 58, 7.0,
    "Training partition of one outer fold\n"
    "(imputed, constant-filtered, min-max scaled, SMOTE-balanced)",
    fc=C_OUT, fs=7.2, bold=True)

box(CX, 128.5, 58, 8.6,
    "STAGE 1   dynamic feature grouping\n"
    "union-find merge of all pairs with |r_ij| > tau;  keep the highest-MI member of\n"
    "each group and every singleton with MI above the median  ->  pool P",
    fc=C_STAGE1, fs=7.0)

box(CX, 117.0, 58, 8.0,
    "INITIALISE   N/2 wolves uniform on {0,1}^|P|,  N/2 bitwise complements (OBL)\n"
    "repair each to |S| >= k_min;  evaluate composite fitness  Eq. (8)",
    fc=C_INIT, fs=7.0)

box(CX, 108.0, 58, 5.4,
    "rank the pack -> leaders alpha, beta, delta;  incumbent g*;  stall <- 0",
    fc=C_INIT, fs=7.0)

box(CX, 96.0, 58, 6.6,
    "update the schedules\n"
    "a(t) = 2 cos(pi t / 2T)   Eq. (6)      z <- 4z(1 - z)   Eq. (4)      k(t) = 1 + 4t/T",
    fs=7.0)

box(CX, 81.0, 56, 6.6,
    "leader-guided move toward alpha, beta, delta\n"
    "A = 2a(t)r1 - a(t),   C' = 2r2[lambda + (1 - lambda)z]   Eq. (1), (2), (7)", fs=7.0)

box(CX, 72.0, 56, 5.4,
    "annealed sigmoid transfer  s_new ~ Bernoulli(sigma(k(t) x_new))   Eq. (3)", fs=7.0)

diamond(CX, 63.0, 30, 7.6, "t > T/2  and\nrand() < 0.15 ?")
box(82.0, 63.0, 26, 6.0, "Levy perturbation\nflip 1 + |Levy(1.5)| bits", fc=C_SIDE, fs=6.9)

box(CX, 53.5, 56, 5.6,
    "repair to |S| >= k_min;   f_new <- Fitness(s_new)   Eq. (8), minimised", fs=7.0)

diamond(CX, 45.0, 26, 7.0, "f_new < f_i ?")
box(82.0, 45.0, 26, 5.0, "accept: wolf i <- s_new", fc=C_SIDE, fs=6.9)

diamond(CX, 35.5, 26, 7.0, "f_new < f(g*) ?")
box(82.0, 35.5, 26, 5.0, "g* <- s_new;  stall <- 0", fc=C_SIDE, fs=6.9)

box(CX, 27.0, 56, 5.6,
    "re-rank the pack; overwrite the worst wolf with g* (elitism);  stall <- stall + 1",
    fs=7.0)

diamond(CX, 17.0, 28, 7.2, "stall >= patience ?")
box(82.0, 17.0, 26, 5.6, "re-initialise the\nworst floor(N/3) wolves;\nstall <- 0",
    fc=C_SIDE, fs=6.6)

diamond(CX, 8.0, 24, 6.8, "t = T ?")

box(CX, -2.0, 58, 7.2,
    "BOUNDED LOCAL SEARCH   at most 2 passes of single-bit flips on g*;\n"
    "accept a flip only if |S| >= k_min and the gain exceeds 1e-4;\n"
    "stop early when a pass accepts nothing", fc=C_STAGE1, fs=7.0)

box(CX, -11.0, 40, 5.4, "Output   S* = decode(g*)", fc=C_OUT, fs=7.6, bold=True)

# ------------------------------------------------------------------ straight edges
for y1, y2 in [(135.0, 132.8), (124.2, 121.0), (113.0, 110.7), (105.3, 99.3),
               (92.7, 84.3), (77.7, 74.7), (69.3, 66.8), (59.2, 56.3),
               (50.7, 48.5), (41.5, 39.0), (32.0, 29.8), (24.2, 20.6),
               (13.4, 11.4), (4.6, 1.6), (-5.6, -8.3)]:
    arrow((CX, y1), (CX, y2))

# ------------------------------------------------------------------ branch edges
for yd, ybox in [(63.0, 63.0), (45.0, 45.0), (35.5, 35.5), (17.0, 17.0)]:
    arrow((CX + 15 if yd == 63.0 else CX + 13 if yd != 17.0 else CX + 14, yd),
          (69.0, ybox))
    tag((CX + 22 if yd == 63.0 else CX + 21), yd + 2.0, "yes", col="#1e8449")

# side boxes rejoin the trunk below
elbow([(82.0, 60.0), (82.0, 57.8), (CX, 57.8)], ls=(0, (4, 2)), col="#7f8c8d")
elbow([(82.0, 42.5), (82.0, 40.2), (CX, 40.2)], ls=(0, (4, 2)), col="#7f8c8d")
elbow([(82.0, 33.0), (82.0, 30.6), (CX, 30.6)], ls=(0, (4, 2)), col="#7f8c8d")
elbow([(82.0, 14.2), (82.0, 11.8), (CX, 11.8)], ls=(0, (4, 2)), col="#7f8c8d")

for yd in (63.0, 45.0, 35.5, 17.0):
    tag(CX + 3.2, yd - 5.0, "no", col="#b03a2e")

# loop back to the top of the main loop
elbow([(CX - 12, 8.0), (6.5, 8.0), (6.5, 99.3), (CX - 29, 99.3)], col="#c0392b")
tag(24.0, 10.2, "no  ->  next iteration", col="#c0392b")
tag(CX + 8.5, 5.4, "yes", col="#1e8449")

ax.set_ylim(-15.5, 143)
plt.tight_layout()
out = os.path.join(FIG, "fig2_workflow.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
