# DF-GWO — parsimonious software defect prediction

Reproduction package for:

> **Parsimonious Software Defect Prediction via a Chaos-Driven Correlation-Grouped Grey
> Wolf Optimizer with Effort-Aware Evaluation**
> Surbhi Syal and Jagdeep Kaur

Every number, table and figure in the paper is produced by the scripts in this
repository. All experiments are deterministic given the seeds recorded in the code.

---

## What is here

DF-GWO is a binary Grey Wolf Optimizer for feature selection in software defect
prediction. It combines correlation-based dynamic feature grouping, a logistic-map
chaotic control schedule, a Lévy-flight perturbation, a composite F1/AUC/MCC fitness
with an explicit size penalty, and three population-management safeguards.

The headline result is **parsimony, not accuracy**, and the code reports it that way:
across twenty PROMISE, AEEEM and ReLink projects DF-GWO selects on average **7.19 of
31.0 metrics (a 76.8 % reduction)**, the best subset-size rank of 17 methods, while its
AUC rank is the lowest of the 17. Both halves of that trade-off are in the tables.

## Requirements

```
python >= 3.9
numpy  2.0.2      pandas 2.3.3      scipy 1.13.1
scikit-learn 1.6.1  joblib 1.5.3    matplotlib 3.9.4
opfunu 1.0.4
```

```bash
pip install -r requirements.txt
```

`opfunu` supplies the official CEC-2022 shift vectors, rotation matrices and shuffle
indices. No transformation data is regenerated here, so the reported errors are on the
same scale as published CEC-2022 results. SMOTE is implemented directly in `common.py`;
there is no `imbalanced-learn` dependency.

## Data

The twenty defect datasets are public but are **not redistributed here**. Obtain them
from their original repositories:

| Repository | Projects | Source |
|---|---|---|
| PROMISE | 12 | <http://promise.site.uottawa.ca/SERepository> |
| AEEEM | 5 | D'Ambros, Lanza & Robbes, *Empir. Softw. Eng.* 17 (2012) |
| ReLink | 3 | Wu, Zhang, Kim & Cheung, *ESEC/FSE* (2011) |

Arrange them as `PROMISE/`, `AEEEM/` and `Relink/` subdirectories and point the code at
their parent:

```bash
export DFGWO_DATA=/path/to/datasets        # Windows: set DFGWO_DATA=C:\path\to\datasets
```

or simply place the three subdirectories in `./data`. Label columns differ by repository
and are handled automatically: PROMISE uses a numeric `bug` count, ReLink an
`isDefective` flag, AEEEM an unnamed trailing buggy/clean column. The lines-of-code
columns used for the effort-aware metrics are `loc`, `CountLineCode` and
`ck_oo_numberOfLinesOfCode` respectively.

## Layout

| File | Role |
|---|---|
| `common.py` | dataset loading, fold preprocessing, SMOTE, predictive and effort-aware metrics |
| `optimizers.py` | every binary optimizer: DF-GWO (with component switches), bGWO1/bGWO2, OBCGWO, SA-bGWO, SR-GWO, bPSO, GA, bWOA, bSSA, BHO, Q2HO-MFTV, EMWS; `Budget` enforces the shared evaluation budget |
| `pipeline.py` | Stage 1 grouping, the Eq. (8) fitness probe, the filter baselines, the method registry and the five-fold driver |
| `cbench.py` | the twelve continuous optimizers — DF-GWO, GWO, I-GWO, RW-GWO, mGWO, CGWO, HO, SHO, PSO, WOA, SCA, GA — and the `FE` counter that enforces one shared evaluation budget |
| `cec2022.py` | the twelve CEC-2022 functions at D ∈ {10, 20}, with class labels and known optima |
| `bench.py` | the classical F1–F13 function definitions |
| `engineering.py` | four constrained design problems with `f` and `g` exposed separately, normalised constraints, the static quadratic penalty and the feasibility check |
| `stats.py` | Friedman, Iman–Davenport, Nemenyi CD, Holm, Wilcoxon, Cliff's delta |
| `analyse.py` / `analyse_v3.py` | build every table (`tables/*.csv`) and figure (`figures/*.png`) |
| `make_fig1.py`, `make_fig_workflow.py` | the pipeline diagram and the algorithm control-flow diagram |

Runners: `run_main.py` (17 methods × 20 datasets), `run_extra.py` (adds a new selector to
cached results), `run_abl.py` (13 ablation configurations), `run_sens.py` (four-factor
sensitivity grid), `run_cec.py` (CEC-2022 at both dimensions, F1–F13, engineering).
`run_bench.py` is retained only to reproduce the continuous experiments of the
earlier revision; `run_cec.py` supersedes it and is what the paper reports.

## Running

```bash
python run_main.py
python run_extra.py
python run_abl.py
python run_sens.py
python run_cec.py
python make_fig1.py
python make_fig_workflow.py
python analyse.py
python analyse_v3.py
```

Each runner writes one JSON per dataset or problem under `results/` and skips work that
is already cached, so an interrupted run resumes. On eight cores the binary experiments
take roughly three hours and `run_cec.py` about four.

`results/`, `tables/` and `figures/` are committed, so the tables and figures in the
paper can be regenerated with `analyse.py` and `analyse_v3.py` alone, without re-running
any experiment.

## Protocol notes

**Binary experiments.** Stratified five-fold CV with a fixed seed; *all* methods see the
identical folds. Median imputation, constant-column removal, min–max scaling and SMOTE
(k = 5) are fitted inside the training partition only; the test fold is touched once, by
the final Random Forest. The fitness probe is an L1-regularised logistic regression on a
held-out 20 % slice of the training fold; `pipeline._fast_f1_auc_mcc` reproduces
scikit-learn's `f1_score`, `roc_auc_score` and `matthews_corrcoef` exactly (verified to
1e-12, including mid-rank tie handling) and is used only to avoid per-call validation
overhead. `optimizers.Budget` counts **every** call against the budget, duplicates
included; the cache only avoids refitting the probe.

Per-method seeds are derived from `zlib.crc32(name)` rather than Python's `hash()`, which
is randomised per process — so repeated runs reproduce the reported numbers exactly.

**Continuous experiments.** Every algorithm stops at exactly the same number of
**objective-function evaluations**, not the same number of iterations. `cbench.FE` raises
`BudgetExhausted` on the call that would exceed the cap, and every optimizer tolerates
that at any point in its loop. This matters: I-GWO scores two candidates per wolf per
iteration and SHO several more, so an iteration-indexed protocol would silently hand them
multiples of the budget given to GWO. All control schedules are indexed by the fraction of
the budget consumed (`FE.p`) rather than by iteration index.

Budgets: 1000 × D on CEC-2022 (10,000 at D = 10 and 20,000 at D = 20) and 15,000 on
F1–F13 and the engineering problems. Population 30, 30 independent runs, fixed seeds.

**Constraint handling.** Engineering constraints are stated in normalised form, so that a
single static penalty coefficient (R = 1e10) is scale-appropriate across constraints
whose natural magnitudes span six orders of magnitude. Feasibility is verified after each
run from the returned design vector (tolerance 1e-4) rather than assumed, and best/mean/
worst statistics are computed over feasible runs only. The four best-known solutions from
the literature reproduce their published objective values under these definitions and are
confirmed feasible.

## License

Code released under the MIT License (see `LICENSE`). The datasets are the property of
their respective repositories and are governed by their own terms.
