# Reproduction Guide: From Training to Filled LaTeX Tables and Figures
> All commands assume your working directory is the project root.

## 1. Run Step 1 — Default Hyperparameters

> **Prerequisites:** conda env `robot_env` active; `src/env.py` patched (Step 0-A).

Submit two job arrays — one for CrossQ (GPU), one for all others (CPU):

```bash
sbatch slurm/step1_crossq_default.sh     # 200 jobs (1 alg × 10 sets × 4 N × 5 seeds)
sbatch slurm/step1_others_default.sh     # 1000 jobs (5 algs × 10 sets × 4 N × 5 seeds)
```

**Wait for all jobs to finish** before proceeding.  Monitor with:
```bash
squeue -u $USER
```

**What this produces:**
```
logs/main_default/{ALG}_N{N}_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
    *.csv  (SB3 training log)
```

---

## 2. Run Step 2 — Optuna Hyperparameter Tuning

Each SLURM job runs **one Optuna trial**. All jobs for the same algorithm share a
single `JournalStorage` log file (NFS-safe, append-only).

```bash
sbatch slurm/step2_tune_hyperparameters.sh   # 300 jobs (6 algs × 50 trials)
```

**What this produces:**
```
logs/best_hyperparams.json             ← consumed by Step 3
logs/optuna_studies/{ALG}_journal.log  ← per-algorithm Optuna log
```
**Wait for all 300 jobs to finish** before proceeding.

---

## 3. Run Step 3 — Tuned Hyperparameters

```bash
sbatch slurm/step3_crossq_tuned.sh       # 200 jobs
sbatch slurm/step3_others_tuned.sh       # 1000 jobs
```

**What this produces:**
```
logs/main_tuned/{ALG}_N{N}_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
```

---

## 4–6. Run Ablation Training — Reward, Observation, Uncertainty

Steps 4, 5, and 6 are handled by a single merged script.
Submit one job array per ablation experiment:

```bash
sbatch --array=0-199 slurm/step4_5_6_ablations.sh ablation_reward       # 200 jobs (4 r_abs × 10 sets × 5 seeds)
sbatch --array=0-199 slurm/step4_5_6_ablations.sh ablation_obs          # 200 jobs (4 obs_abs × 10 sets × 5 seeds)
sbatch --array=0-199 slurm/step4_5_6_ablations.sh ablation_uncertainty  # 200 jobs (4 noise × 10 sets × 5 seeds)
```

**What this produces:**
```
logs/ablation_reward_{CONDITION}/CrossQ_N3_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz          ← reward + episode-length data

logs/ablation_obs_{OBS_MODE}/CrossQ_N3_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz

logs/ablation_uncertainty_{MODE}/CrossQ_N3_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
```

> **Note — reward ablation metric:** `analyze_results.py` reads `ep_lengths`
> from the training `evaluations.npz` files and reports the **mean episode
> length** (at the best-reward checkpoint) for each reward-ablation condition
> instead of the IQM of episode reward.  Shorter episodes indicate faster task
> completion, which is the meaningful signal when reward terms are removed.
> No separate 50-episode evaluation is required for reward or obs ablations.

---

## 7. Run Step 7 — Domain Randomization

```bash
sbatch slurm/step7_dr.sh           # 600 jobs (3 DR modes × 10 sets × 4 N × 5 seeds)
```

---

## 8. Post-Training Evaluation

- **Uncertainty ablation (Table 5):** The cross-evaluation matrix evaluates
  each trained policy under all four noise conditions (not just the one it
  trained on), so 4 train × 4 eval combinations are needed.
- **Domain randomization (Table 7):** Policies must be evaluated under
  controlled in-distribution and out-of-distribution wind ranges.

All other tables (main results, reward ablation, obs ablation) are filled
directly from the training NPZ files — no separate eval step is required.

> **Re-submission safety:** `evaluate.py` checks whether a matching row already
> exists in the output CSV before running any episodes (matched on algorithm,
> experiment, ablation, num_robots, env_set, seed, eval_wind_min,
> eval_wind_max, and eval_uncertainty_mode).  If found it prints `[SKIP]` and
> exits immediately — so re-submitting failed array indices never produces
> duplicate rows in the CSV.

### 8-A. Reward ablation (Table 3)

`eval_ablations.sh` runs 50-episode rollouts to capture terminal-condition
statistics (`sprayed_pct`, `collision_pct`, `max_steps_pct`) that the training
NPZ files do not store.  `analyze_results.py` prefers this CSV over the NPZ
fallback when it exists.

> **Note — obs ablation:** Table 4 does **not** need a separate eval step.
> `analyze_results.py` reads the training `evaluations.npz` files directly
> for that experiment.  Skip straight to Step 9 once Steps 4–6 training is done.

```bash
sbatch --array=0-199 slurm/eval_ablations.sh ablation_reward
```

After all jobs finish, merge the per-job CSVs:

```bash
OUT="logs/results/ablation_reward.csv"
first_file=$(ls logs/results/tmp/ablation_reward/*/set*/result_*.csv | head -n 1)
head -n 1 "$first_file" > "$OUT"
tail -n +2 -q logs/results/tmp/ablation_reward/*/set*/result_*.csv >> "$OUT"
echo "Merged → $OUT"
```

### 8-B. Uncertainty ablation (Table 5)


The uncertainty ablation requires a cross-evaluation matrix
(4 train conditions × 4 eval conditions) and is evaluated across
**all 10 env sets**:

```bash
# 4 train × 4 eval × 10 sets × 5 seeds = 800 jobs
sbatch --array=0-799 slurm/eval_ablations.sh ablation_uncertainty
```

After all uncertainty jobs finish, merge the per-job CSVs:

```bash
OUT="logs/results/ablation_uncertainty.csv"
first_file=$(ls logs/results/tmp/ablation_uncertainty/train_*_eval_*/set*/result_*.csv | head -n 1)
head -n 1 "$first_file" > "$OUT"
tail -n +2 -q logs/results/tmp/ablation_uncertainty/train_*_eval_*/set*/result_*.csv >> "$OUT"
echo "Merged → $OUT"
```

### 8-C. Domain randomization (Table 7)

```bash
sbatch --array=0-599 slurm/eval_ablations.sh dr
```

Merge in-distribution:
```bash
OUT="logs/results/dr_inDist.csv"
first_file=$(ls logs/results/tmp/dr/*/*/inDist_*.csv | head -n 1)
head -n 1 "$first_file" > "$OUT"
tail -n +2 -q logs/results/tmp/dr/*/*/inDist_*.csv >> "$OUT"
echo "Merged → $OUT"
```

Merge OOD:
```bash
OUT="logs/results/dr_OOD.csv"
first_file=$(ls logs/results/tmp/dr/*/*/OOD_*.csv | head -n 1)
head -n 1 "$first_file" > "$OUT"
tail -n +2 -q logs/results/tmp/dr/*/*/OOD_*.csv >> "$OUT"
echo "Merged → $OUT"
```

### 8-D. Wind sweep for Figure 4

```bash
sbatch slurm/eval_wind_sweep.sh          # 100 jobs (2 DR × 10 bins × 5 seeds)
```

> `eval_wind_sweep.sh` appends directly to `results/wind_sweep.csv`
> (all 100 jobs share the same file, guarded by `fcntl` file locking in
> `evaluate.py`).  No merge step is needed.

**Sanity check** — expected line counts (header + one row per job):

```bash
wc -l logs/results/ablation_reward.csv       # 201  (200 runs + header)
wc -l logs/results/ablation_uncertainty.csv  # 801  (800 runs + header)
wc -l logs/results/dr_inDist.csv             # 601  (600 runs + header)
wc -l logs/results/dr_OOD.csv                # 601  (600 runs + header)
wc -l logs/results/wind_sweep.csv            # 101  (100 runs + header)
```

> **No separate CSVs for main results or obs ablations** — those are
> read from training NPZs by `analyze_results.py`.

---

## 8-E. (Optional) Hyperparameter Sensitivity — Table 6

> This step is **independent** of Steps 1–8D and can be run at any time
> after Step 2 has produced `logs/best_hyperparams.json`.

The sensitivity sweep trains each algorithm over a 7-point grid for each of
its tunable hyperparameters and computes the coefficient of variation (CV) of
IQM across the grid.  Each SLURM job handles one complete algorithm.

```bash
# 6 jobs (one per algorithm) — array=0-5 is declared in the script
sbatch slurm/step8_sensitivity_hp.sh
```

**What this produces:**

```
logs/results/cv_table.csv                 ← one row per (alg, HP)
logs/results/sensitivity_hp_latex_rows.txt ← ready-to-paste LaTeX rows for tab:sensitivity_hp
```

After all six jobs finish, regenerate the combined LaTeX table:

```bash
python sensitivity_hp.py --write_latex_only --results_dir logs/results
```

> **Note:** Each algorithm job is independent — if one fails, re-submit only
> that index (e.g., `sbatch --array=2 slurm/step8_sensitivity_hp.sh` for PPO).
> Results are appended to `cv_table.csv` so re-running a failed job is safe.

> ⚠️ **Known issue in `step8_sensitivity_hp.sh`:** the script currently
> invokes Python via `conda run -n robot_env python3 ...`.  With
> `#SBATCH --export=NONE`, `conda` is not on PATH and the job will fail.
> Replace line 41 with the full interpreter path:
> `/homes/choton/miniconda3/envs/robot_env/bin/python3 sensitivity_hp.py \`

---

## 9. Aggregate Results and Generate Table CSVs

Once **all** evaluation jobs are complete:

```bash
python analyze_results.py --log_root logs --results_dir logs/results
```

**What this produces in `results/`:**

| File | Used for | Primary metric |
|---|---|---|
| `main_default_summary.csv` | Tab. 1 data | Mean ± Std reward, IQM |
| `main_tuned_summary.csv`   | Tab. 2 data | Mean ± Std reward, IQM |
| `ablation_reward_agg.csv`  | Tab. 3 data | **Mean episode length ± Std** |
| `ablation_obs_agg.csv`     | Tab. 4 data | IQM of reward, Δ IQM % |
| `ablation_uncertainty_agg.csv` | Tab. 5 data | IQM (same / full / det.) |
| `dr_results_agg.csv`       | Tab. 7 data | In-dist IQM, OOD IQM, CVaR |
| `*_latex_rows.txt`         | **Ready-to-paste LaTeX rows** | — |

> **Data sources:** Main results (Tables 1 & 2) and observation ablation
> (Table 4) are read **directly from training `evaluations.npz` files** in
> `--log_root`.  Reward ablation (Table 3) is read from
> `logs/results/ablation_reward.csv` (produced by `eval_ablations.sh`), with
> automatic NPZ fallback if the CSV does not exist.  Uncertainty ablation
> (Table 5) and domain randomization (Table 7) are read from **evaluation
> CSVs** in `--results_dir` produced by `eval_ablations.sh`.

> **Reward ablation note:** `ablation_reward_agg.csv` and
> `ablation_reward_latex_rows.txt` report **mean episode length** (not IQM of
> reward).  The LaTeX row format is:
> ```
> Condition & Removed terms & Mean reward & Std & Mean ep. length ± Std \\
> ```
> The condition with the **shortest** mean episode length is bolded (fastest
> task completion).

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

Each row now reports **mean episode length ± std** as the primary metric
(shorter = faster task completion = better).  Example cell:
```
Full reward & --- & $910.2$ & $38.4$ & $\mathbf{412.3 \pm 21.1}$ \\
```

### Table 4 (observation ablation)

1. Open `results/ablation_obs_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_obs`

### Table 5 (uncertainty ablation)

1. Open `results/ablation_uncertainty_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_uncertainty`

### Table 6 (hyperparameter sensitivity)

This table (CV values) requires a dedicated sensitivity sweep. The values
come from varying each hyperparameter over a 7-point grid.  Run via SLURM
(see **Step 8-E** above) or manually:

```bash
python sensitivity_hp.py --algorithm TRPO   --results_dir logs/results
python sensitivity_hp.py --algorithm CrossQ --results_dir logs/results
python sensitivity_hp.py --algorithm PPO    --results_dir logs/results
python sensitivity_hp.py --algorithm A2C    --results_dir logs/results
python sensitivity_hp.py --algorithm TQC    --results_dir logs/results
python sensitivity_hp.py --algorithm ARS    --results_dir logs/results
```

After all six finish, regenerate the LaTeX rows only:
```bash
python sensitivity_hp.py --write_latex_only --results_dir logs/results
```

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
| 2  | step2_tune_hyperparameters.sh | 300 |
| 3a | step3_crossq_tuned.sh   | 200  |
| 3b | step3_others_tuned.sh   | 1000 |
| 4–6a | step4_5_6_ablations.sh (reward)      | 200 |
| 4–6b | step4_5_6_ablations.sh (obs)         | 200 |
| 4–6c | step4_5_6_ablations.sh (uncertainty) | 200 |
| 7  | step7_dr.sh             | 600  |
| 8a | eval_ablations.sh (ablation_reward) | 200 |
| 8b | eval_ablations.sh (uncertainty) — **all 10 sets** | **800** |
| 8c | eval_ablations.sh (dr)          | 600  |
| 8d | eval_wind_sweep.sh      | 100  |
| **Total** | | **5600** |
| 8e *(optional)* | step8_sensitivity_hp.sh | 6 |

> **Step 2:** 300 jobs = 6 algorithms × 50 Optuna trials, one SLURM job per trial
> (the `--array=0-299` is declared inside the script; just `sbatch` the file).
>
> **Compared to the original version:** `eval_main.sh` jobs (2 × 1200 = 2400)
> removed entirely — main results now read from training NPZs.
> Obs ablation eval jobs (200) also removed (reads NPZ directly).
> Uncertainty ablation eval jobs increased from 80 → 800 (all 10 sets).
> Net saving vs original: **~2000 fewer jobs** (excl. optional step 8e).

---

## Dependency Graph (submit in order)

```
Step 1 (default train)
    └─► analyze_results.py → Tab. 1   (reads NPZ directly — no separate eval)
        plot_figures.py → Fig. 1

Step 2 (tune) ─► Step 3 (tuned train)
                    └─► analyze_results.py → Tab. 2   (reads NPZ directly)
                        plot_figures.py → Fig. 2, 3

Steps 4–6a (step4_5_6_ablations.sh reward)
    └─► eval_ablations.sh (ablation_reward, 200 jobs)
            └─► merge CSVs → logs/results/ablation_reward.csv
                    └─► analyze_results.py → Tab. 3   (CSV preferred; NPZ fallback)

Steps 4–6b (step4_5_6_ablations.sh obs)
    └─► analyze_results.py → Tab. 4   (reads NPZ directly — NO eval step)

Steps 4–6c (step4_5_6_ablations.sh uncertainty)
    └─► eval_ablations.sh (uncertainty, 800 jobs, all 10 sets)
            └─► merge CSVs → logs/results/ablation_uncertainty.csv
                    └─► analyze_results.py → Tab. 5

Step 7 ─► eval_ablations.sh (dr, 600 jobs)
              └─► merge CSVs → logs/results/dr_inDist.csv + logs/results/dr_OOD.csv
                      └─► analyze_results.py → Tab. 7
       └─► eval_wind_sweep.sh → logs/results/wind_sweep.csv
                                    └─► plot_figures.py → Fig. 4, 5

Step 2 ─► (optional) step8_sensitivity_hp.sh (6 jobs)
              └─► python sensitivity_hp.py --write_latex_only → Tab. 6
```