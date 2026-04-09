# Reproduction Guide: From Training to Filled LaTeX Tables and Figures

> All commands assume your working directory is the project root:
> `/homes/choton/rl4pag/selective_spraying_using_RL/`

---

## 0. Pre-flight Checks and Bug Fixes

### 0-A. Replace env.py (critical bug fix)
The original `src/env.py` has a `NameError`: the variable `remaining` is used
in the `no_cov` branch of `step()` before it is assigned, and also in the
`info` dict. This causes a crash whenever `reward_ablation="no_cov"`.

```bash
cp env_fixed.py src/env.py
```

### 0-B. Replace the Step-1 SLURM scripts (5-seed fix)
The original Step-1 scripts only ran **1 seed** but the paper tables require
**5 seeds**. Use the fixed versions:

```bash
cp step1_crossq_default_fixed.sh step1_crossq_default.sh
cp step1_others_default_fixed.sh step1_others_default.sh
```

### 0-C. Create output directories

```bash
mkdir -p slurm_outputs results figures logs
```

### 0-D. Place new scripts in the project root

Copy these new scripts from the delivery folder to the project root:
```
evaluate.py
analyze_results.py
plot_figures.py
eval_main.sh
eval_ablations.sh
eval_wind_sweep.sh
```

---

## 1. Run Step 1 — Default Hyperparameters

> **Prerequisites:** conda env `rl4pag` active; `src/env.py` patched (Step 0-A).

Submit two job arrays — one for CrossQ (GPU), one for all others (CPU):

```bash
sbatch step1_crossq_default.sh     # 200 jobs (1 alg × 10 sets × 4 N × 5 seeds)
sbatch step1_others_default.sh     # 1000 jobs (5 algs × 10 sets × 4 N × 5 seeds)
```

**Wait for all jobs to finish** before proceeding.  Monitor with:
```bash
squeue -u $USER
```

**What this produces:**
```
logs/*_main_default/{ALG}_N{N}_env{SET}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
    *.csv  (SB3 training log)
```

---

## 2. Run Step 2 — Optuna Hyperparameter Tuning

```bash
sbatch step2_tune_hyp.sh           # 6 jobs (one per algorithm)
```

**What this produces:**
```
logs/best_hyperparams.json         ← consumed by Step 3
logs/step2_tune/optuna_{ALG}.csv   ← per-algorithm trial history
```

---

## 3. Run Step 3 — Tuned Hyperparameters

```bash
sbatch step3_crossq_tuned.sh       # 200 jobs
sbatch step3_others_tuned.sh       # 1000 jobs
```

**What this produces:**
```
logs/*_main_tuned/{ALG}_N{N}_env{SET}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
```

---

## 4. Run Step 4 — Reward Ablation

```bash
sbatch step4_ablation_reward.sh    # 20 jobs (4 conditions × 5 seeds)
```

---

## 5. Run Step 5 — Observation Ablation

```bash
sbatch step5_ablation_obs.sh       # 25 jobs (5 conditions × 5 seeds)
```

---

## 6. Run Step 6 — Uncertainty Ablation

```bash
sbatch step6_ablation_uncertainty.sh  # 20 jobs (4 conditions × 5 seeds)
```

---

## 7. Run Step 7 — Domain Randomization

```bash
sbatch step7_dr.sh                 # 600 jobs (3 DR modes × 10 sets × 4 N × 5 seeds)
```

---

## 8. Post-Training Evaluation (50 episodes per policy)

The SB3 `EvalCallback` during training only runs 5 episodes per checkpoint.
For the final tables the paper requires **50 evaluation episodes per run**.
Run `evaluate.py` via the evaluation SLURM scripts.

### 8-A. Main results (Tables 1 & 2)

```bash
# Default HPs (Steps 1 → Table 1)
# Edit eval_main.sh to set --array=0-299 (6 algs × 10 sets × 4 N × ...).
# Simpler: run interactively for default HP (faster, fewer jobs):
sbatch --array=0-299 eval_main.sh default

# Tuned HPs (Steps 3 → Table 2)
sbatch --array=0-1199 eval_main.sh tuned
```

> **Tip:** Because file-append is used, if individual jobs fail you can
> simply re-submit the failed array indices.

### 8-B. Ablation experiments (Tables 3, 4, 5)

```bash
sbatch --array=0-19  eval_ablations.sh ablation_reward
sbatch --array=0-24  eval_ablations.sh ablation_obs
# For uncertainty, cross-eval requires 4 train × 4 eval × 5 seeds = 80 jobs:
sbatch --array=0-79  eval_ablations.sh ablation_uncertainty
```

### 8-C. Domain randomization (Table 7)

```bash
sbatch --array=0-599 eval_ablations.sh dr
```

### 8-D. Wind sweep for Figure 4

```bash
sbatch eval_wind_sweep.sh          # 100 jobs (2 DR × 10 bins × 5 seeds)
```

---

## 9. Aggregate Results and Generate Table CSVs

Once **all** evaluation jobs are complete:

```bash
python analyze_results.py --results_dir results
```

**What this produces in `results/`:**
| File | Used for |
|---|---|
| `main_default_summary.csv` | Tab. 1 data |
| `main_tuned_summary.csv`   | Tab. 2 data |
| `ablation_reward_agg.csv`  | Tab. 3 data |
| `ablation_obs_agg.csv`     | Tab. 4 data |
| `ablation_uncertainty_agg.csv` | Tab. 5 data |
| `dr_results_agg.csv`       | Tab. 7 data |
| `*_latex_rows.txt`         | **Ready-to-paste LaTeX rows** |

---

## 10. Generate All Figures

```bash
python plot_figures.py \
    --log_root    logs \
    --results_dir results \
    --figures_dir figures
```

**Figures produced:**

| File | LaTeX label | Description |
|---|---|---|
| `figures/default_learning_curves.png` | `fig:default_hyp` | Learning curves (default HPs, all N) |
| `figures/default_{N}_robots.png`      | (per-N version)   | Individual N panels |
| `figures/random_learning_curves.png`  | `fig:random_hyp`  | Learning curves (tuned HPs) |
| `figures/random_{N}_robots.png`       | (per-N version)   | |
| `figures/scalability.png`             | `fig:scalability` | IQM + ep-length vs N |
| `figures/wind_sensitivity.png`        | `fig:wind_sensitivity` | IQM vs wind speed |
| `figures/dr_curves.png`               | `fig:dr_curves`   | DR training curves |

---

## 11. Fill LaTeX Tables

### Table 1 & 2 (main results)

1. Open `results/main_default_latex_rows.txt`
2. Each line is a `Algorithm & N=2 & N=3 & N=4 & N=5 \\` row
3. Replace the dummy rows in `full_experiments.tex` between `\midrule` and `\bottomrule` of `tab:default_hyp`
4. Repeat with `main_tuned_latex_rows.txt` for `tab:random_hyp`

Example format of each cell:
```
$850.3 \pm 42.1\ (847.2)^\dagger$
```

### Table 3 (reward ablation)

1. Open `results/ablation_reward_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_reward`

### Table 4 (observation ablation)

1. Open `results/ablation_obs_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_obs`

### Table 5 (uncertainty ablation)

1. Open `results/ablation_uncertainty_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_uncertainty`

### Table 6 (hyperparameter sensitivity)

This table (CV values) requires a dedicated sensitivity sweep. The values
come from varying each hyperparameter over a 7-point grid.  Run manually:

```bash
python sensitivity_hp.py --algorithm TRPO  --results_dir results
python sensitivity_hp.py --algorithm CrossQ --results_dir results
python sensitivity_hp.py --algorithm PPO   --results_dir results
python sensitivity_hp.py --algorithm A2C   --results_dir results
python sensitivity_hp.py --algorithm TQC   --results_dir results
# ARS: only alpha, no gae
python sensitivity_hp.py --algorithm ARS   --results_dir results
```

(See `sensitivity_hp.py` for this script.)

### Table 7 (DR results)

1. Open `results/dr_results_latex_rows.txt`
2. Replace dummy rows in `tab:dr_results`

### Table 8 (observation gap, sim-to-real)

This table comes from **manual CoppeliaSim inference experiments** (as noted
in the tex comment). Run the trained CrossQ+full-DR policy inside CoppeliaSim
with:
- GPS noise σ = 0.05 m
- Wind estimate latency = 1 step
- 5% packet loss

Record IQM from 50 episodes per condition and fill manually.

### Table 9 (scalability / compute)

Run:
```bash
python measure_compute.py --log_root logs --results_dir results
```

(See `measure_compute.py`.) This measures wall-clock time per training run
from log timestamps, and inference latency from a timed loop.

---

## 12. Update Figure Captions

After generating figures, update the captions and `\includegraphics` paths in
`full_experiments.tex` to point to the correct figure files, e.g.:

```latex
\includegraphics[width=\linewidth]{figures/default_learning_curves.png}
```

---

## Summary of Job Counts

| Step | Script | Jobs |
|---|---|---|
| 1a | step1_crossq_default.sh | 200 |
| 1b | step1_others_default.sh | 1000 |
| 2  | step2_tune_hyp.sh       | 6    |
| 3a | step3_crossq_tuned.sh   | 200  |
| 3b | step3_others_tuned.sh   | 1000 |
| 4  | step4_ablation_reward.sh| 20   |
| 5  | step5_ablation_obs.sh   | 25   |
| 6  | step6_ablation_uncertainty.sh | 20 |
| 7  | step7_dr.sh             | 600  |
| 8a | eval_main.sh (default)  | 300  |
| 8b | eval_main.sh (tuned)    | 1200 |
| 8c | eval_ablations.sh (reward) | 20 |
| 8d | eval_ablations.sh (obs) | 25   |
| 8e | eval_ablations.sh (uncertainty) | 80 |
| 8f | eval_ablations.sh (dr)  | 600  |
| 8g | eval_wind_sweep.sh      | 100  |
| **Total** | | **~4396** |

---

## Dependency Graph (submit in order)

```
Step 1 (default train)
    └─► eval_main.sh (default)
            └─► analyze_results.py → Tab. 1
                plot_figures.py → Fig. 1

Step 2 (tune) ─► Step 3 (tuned train)
                    └─► eval_main.sh (tuned)
                            └─► analyze_results.py → Tab. 2
                                plot_figures.py → Fig. 2, 3

Step 4 ─► eval_ablations.sh (reward) ─► analyze_results.py → Tab. 3
Step 5 ─► eval_ablations.sh (obs)    ─► analyze_results.py → Tab. 4
Step 6 ─► eval_ablations.sh (uncertainty) ─► analyze_results.py → Tab. 5

Step 7 ─► eval_ablations.sh (dr)      ─► analyze_results.py → Tab. 7
       └─► eval_wind_sweep.sh         ─► plot_figures.py → Fig. 4
                                      ─► plot_figures.py → Fig. 5
```
