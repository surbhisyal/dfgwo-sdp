"""Figure 1: the DF-GWO experimental pipeline.

Laid out on an explicit grid. Every box is a `Box` that knows its own edges, and every
arrow is anchored to those edges rather than to hand-typed coordinates, so boxes cannot
overlap and arrows cannot start or end in mid-air. All connections are either straight
(when two boxes share a centre line) or right-angled elbows; nothing runs diagonally
across the figure.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

C_DATA = "#d6eaf8"      # data and protocol
C_PREP = "#d5f5e3"      # training-partition processing
C_STAGE = "#fdebd0"     # DF-GWO stages
C_TEST = "#fadbd8"      # anything that touches the held-out fold
C_SIDE = "#ffffff"
C_METRIC = "#eaeded"
EC = "#37474f"

fig, ax = plt.subplots(figsize=(12.6, 10.2))
ax.set_xlim(-3, 124)
ax.set_ylim(-33, 71)
ax.axis("off")


class Box:
    """A rounded box that knows where its own edges are."""

    def __init__(self, cx, cy, w, h, text, fc=C_SIDE, fs=8.4, bold=False, ec=EC):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.55,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=1.1, zorder=3))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
                weight="bold" if bold else "normal", linespacing=1.5)

    # edge anchors, with the rounded-corner padding taken into account
    @property
    def left(self):
        return self.cx - self.w / 2 - 0.55

    @property
    def right(self):
        return self.cx + self.w / 2 + 0.55

    @property
    def top(self):
        return self.cy + self.h / 2 + 0.55

    @property
    def bottom(self):
        return self.cy - self.h / 2 - 0.55


def arrow(p, q, ls="-", col=EC, lw=1.15, head=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>" if head else "-",
                                 mutation_scale=12, lw=lw, color=col,
                                 linestyle=ls, zorder=2, shrinkA=0, shrinkB=0))


def across(a, b, ls="-", col=EC):
    """Straight horizontal arrow, a.right -> b.left (boxes share a centre line)."""
    arrow((a.right, a.cy), (b.left, b.cy), ls=ls, col=col)


def down(a, b, ls="-", col=EC):
    """Straight vertical arrow, a.bottom -> b.top (boxes share a centre line)."""
    arrow((a.cx, a.bottom), (b.cx, b.top), ls=ls, col=col)


def elbow(pts, ls="-", col=EC, lw=1.15):
    """Right-angled polyline; only the final segment carries the arrowhead."""
    for p, q in zip(pts[:-2], pts[1:-1]):
        arrow(p, q, ls=ls, col=col, lw=lw, head=False)
    arrow(pts[-2], pts[-1], ls=ls, col=col, lw=lw)


def label(x, y, s, col="#5d6d7e", fs=7.4, rot=0):
    ax.text(x, y, s, fontsize=fs, color=col, ha="center", va="center",
            style="italic", zorder=5, rotation=rot,
            bbox=dict(fc="white", ec="none", pad=1.2))


# ===================================================================== protocol band
A1 = Box(18, 63, 32, 11,
         "Twenty projects\nPROMISE / AEEEM / ReLink\n(n = 56-1862,  d = 20-61)", C_DATA)
A2 = Box(62, 63, 34, 11,
         "Stratified 5-fold cross-validation\n(outer folds fixed and shared\nby every method)",
         C_DATA)
A3 = Box(104, 63, 26, 11,
         "Held-out test fold\n(untouched until the\nfinal Random Forest)", C_TEST)

A4 = Box(62, 46, 44, 10,
         "Training partition only\nmedian impute  |  drop constant columns\n"
         "min-max scale  |  SMOTE (k = 5)", C_PREP)

across(A1, A2)
across(A2, A3)
label((A2.right + A3.left) / 2, 66.2, "test partition")
down(A2, A4)
label(62, 54.2, "training partition")

# ===================================================================== stage 1
S1 = Box(62, 29, 56, 12,
         "STAGE 1   dynamic feature grouping\n"
         "Pearson |rho| > tau = 0.85, union-find merge;  highest-MI member of each\n"
         "group plus every high-MI singleton   ->   candidate pool P", C_STAGE)
down(A4, S1)

# ===================================================================== main loop
LOOP_L, LOOP_R, LOOP_B, LOOP_T = 6, 70, -14, 15
ax.add_patch(Rectangle((LOOP_L, LOOP_B), LOOP_R - LOOP_L, LOOP_T - LOOP_B,
                       fc="#fdf2f2", ec="#c0392b", lw=1.2, linestyle=(0, (5, 3)),
                       zorder=0))
ax.text(LOOP_L + 2, LOOP_T - 2.4, "Main loop   t = 1 ... T = 30", fontsize=9,
        weight="bold", color="#922b21", ha="left", va="center", zorder=5)

S2 = Box(38, 6, 56, 8,
         "STAGE 2   chaos-driven search\n"
         "cosine a(t) Eq. (6)  |  logistic C' Eq. (7)  |  Levy flight (t > T/2)", fs=8.2)
S3 = Box(38, -6, 56, 10,
         "STAGE 3   composite fitness  Eq. (8)\n"
         "0.40 F1 + 0.40 AUC + 0.20 MCC* + gamma |S| / |P|\n"
         "on an internal 20% validation slice (L1-logistic probe)", fs=8.2)
down(S2, S3)

# Stage 1 feeds the loop: down the spine, then left into the top of Stage 2.
elbow([(S1.cx, S1.bottom), (S1.cx, 18.5), (S2.cx, 18.5), (S2.cx, S2.top)])

# ===================================================================== stage 4
P4_L, P4_R = 75, 115
ax.add_patch(Rectangle((P4_L, LOOP_B), P4_R - P4_L, LOOP_T - LOOP_B,
                       fc="#f5eef8", ec="#6c3483", lw=1.2, linestyle=(0, (5, 3)),
                       zorder=0))
ax.text((P4_L + P4_R) / 2, LOOP_T - 2.4, "STAGE 4   population management",
        fontsize=9, weight="bold", color="#5b2c6f", ha="center", va="center", zorder=5)

G1 = Box(95, 6, 34, 4.4, "Opposition-based initialisation (t = 0)", fs=8.0)
G2 = Box(95, 0, 34, 4.4, "Adaptive restart after 5 stalled iterations", fs=8.0)
G3 = Box(95, -6, 34, 4.4, "Bounded local search (2 passes, after the loop)", fs=8.0)

arrow((S2.right, S2.cy), (G1.left, G1.cy))
arrow((G3.left, G3.cy), (S3.right, S3.cy))

# ===================================================================== output chain
SS = Box(38, -20.5, 40, 6, "Selected subset  S*     (|S*| >= 3)", C_PREP, bold=True)
down(S3, SS)

RF = Box(18, -35, 32, 7,
         "Random Forest scorer\n100 trees, balanced class weights", C_DATA)
PR = Box(84, -35, 40, 7,
         "Predicted defect probabilities\non the held-out test fold", C_TEST)

elbow([(SS.cx, SS.bottom), (SS.cx, -27.0), (RF.cx, -27.0), (RF.cx, RF.top)])
across(RF, PR)

# the held-out fold travels down the right-hand rail and meets the scorer's output
RAIL = 119.5
elbow([(A3.cx, A3.bottom), (A3.cx, 53.5), (RAIL, 53.5), (RAIL, PR.cy), (PR.right, PR.cy)],
      ls=(0, (5, 3)), col="#b9770e")
label(RAIL, 24, "held-out fold", col="#b9770e", rot=90)

# ===================================================================== metrics
M1 = Box(30, -49, 46, 7,
         "Predictive metrics\nAccuracy  |  F1  |  AUC  |  MCC  |  G-mean", C_METRIC)
M2 = Box(88, -49, 50, 7,
         "Effort-aware metrics (defect-density ranking)\n"
         "Recall@20%LOC  |  Popt@20%  |  EA-AUC", C_METRIC)

elbow([(PR.cx, PR.bottom), (PR.cx, -42.5), (M1.cx, -42.5), (M1.cx, M1.top)])
elbow([(PR.cx, PR.bottom), (PR.cx, -42.5), (M2.cx, -42.5), (M2.cx, M2.top)])

ax.set_ylim(-55, 70)
plt.tight_layout()
out = os.path.join(FIG, "fig1_pipeline.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
