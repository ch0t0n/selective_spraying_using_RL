# Learning to Spray in an Uncertain and Windy Environment

This is the codebase for the paper "Learning to Spray in an Uncertain and Windy Environment". In this paper, we present a Reinforcement Learning (RL) solution for multi-robot precision spraying.  

## Setup  

It is recommended to run this codebase on Linux. The necessary packages and libraries needed to run the code are provided in the `requirements.txt` file. Run the following command to install the required packages: 

```
pip install -r requirements.txt
```

# Summary (TLDR)

Please follow the instructions from **Part A** if you want to run the experiments on a local machine (which may take a long time). Follow only **Part B** if you have access to an HPC Cluster. **If you want a single file solution of our method, please check the python script `single_file/final_spraying_env_v1.py` which contains the code for training our environment on 10 different variations using CrossQ algorithm.** We tested the single file implementation on a desktop with AMD Ryzen 7 CPU, RTX 3060 GPU, and 16 GB RAM. It took 2 hours per variation (20 hours in total) for training. Once the environment is trained, please follow the instructions from **Simulation** near the end of this document to get instructions on how to run simulation on CoppeliaSim using the trained models. 

# Part A — Local Machine (No HPC Cluster)

This section covers how to run every experiment step on a **single local machine**
without SLURM or any job scheduler.  All commands call `python` directly.

> **Device note:** Replace `--device cpu` with `--device cuda` anywhere below
> if you have a CUDA-capable GPU. CrossQ and TQC benefit the most from a GPU.

> **Full reproduction time warning:** The full experiment grid (5 seeds × 10 env
> sets × 4 robot counts) takes hundreds of CPU-hours per algorithm.  The commands
> below show both a **minimal single-run** (one seed, one env set) for sanity
> checking and **loop variants** for fuller coverage.  Prioritise the single-run
> first; expand to loops only after confirming everything works end-to-end.

---

## A-1. Step 1 — Default Hyperparameters (Local)

### Minimal single run (quick sanity check)

```bash
# CrossQ, env set 1, 3 robots, seed 42, 2 M steps
python train.py \
    --algorithm   CrossQ \
    --set         1 \
    --num_robots  3 \
    --seed        42 \
    --steps       2000000 \
    --experiment  main \
    --device      cpu \
    --verbose     1 \
    --log_steps   10000
```

Outputs land in `logs/main_default/CrossQ_N3_env1_seed42/`.

### Run all algorithms for one env set and one seed

```bash
for ALG in CrossQ TQC PPO A2C TRPO ARS; do
    python train.py \
        --algorithm   $ALG \
        --set         1 \
        --num_robots  3 \
        --seed        42 \
        --steps       2000000 \
        --experiment  main \
        --device      cpu \
        --verbose     1 \
        --log_steps   10000
done
```

### Expand to multiple seeds / env sets (optional, time-consuming)

```bash
for ALG in CrossQ TQC PPO A2C TRPO ARS; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      for N in 2 3 4 5; do
        python train.py \
            --algorithm   $ALG \
            --set         $SET \
            --num_robots  $N \
            --seed        $SEED \
            --steps       2000000 \
            --experiment  main \
            --device      cpu \
            --verbose     1 \
            --log_steps   10000
      done
    done
  done
done
```

---

## A-2. Step 2 — Transfer Learning (Local)

Step 2 fine-tunes from the Step 1 Set 1 default-policy checkpoint for the
same algorithm, robot count, and seed.  Targets are Sets 2–10.

### Minimal single run

```bash
python train.py \
    --algorithm      CrossQ \
    --set            2 \
    --num_robots     3 \
    --seed           42 \
    --steps          2000000 \
    --experiment     main \
    --transfer_from  logs/main_default/CrossQ_N3_env1_seed42/best_model/best_model.zip \
    --device         cpu \
    --verbose        1 \
    --log_steps      10000
```

Outputs land in `logs/main_transfer/CrossQ_N3_env2_seed42/`.

### Loop over all transfer targets

```bash
for ALG in CrossQ TQC PPO A2C TRPO ARS; do
  for SET in 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      for N in 2 3 4 5; do
        SRC="logs/main_default/${ALG}_N${N}_env1_seed${SEED}/best_model/best_model.zip"
        python train.py \
            --algorithm      $ALG \
            --set            $SET \
            --num_robots     $N \
            --seed           $SEED \
            --steps          2000000 \
            --experiment     main \
            --transfer_from  $SRC \
            --device         cpu \
            --verbose        1 \
            --log_steps      10000
      done
    done
  done
done
```

---

## A-3. Step 3 — Hyperparameter Tuning (Local)

On a local machine, run all trials for one algorithm in a **single process** by
setting `--n_trials` to the number of trials you want.  The `JournalStorage`
log file is created automatically; there is no need for shared-filesystem
workarounds when running on one machine.

```bash
mkdir -p logs/optuna_studies

# Tune each algorithm — 50 trials each (matches HPC budget)
for ALG in A2C ARS PPO TRPO CrossQ TQC; do
    python tune.py \
        --algorithm   $ALG \
        --device      cpu \
        --n_trials    50 \
        --tune_steps  500000 \
        --storage     logs/optuna_studies/${ALG}_journal.log \
        --study_name  ${ALG}_tune \
        --output_json logs/best_hyperparams.json
done
```

> **Tip:** `--tune_steps 500000` (500 k) is used above to keep wall-clock time
> manageable.  The HPC scripts use 2 M steps per trial.  Lower values are
> faster but may produce noisier estimates.  Increase if time permits.

> **Resuming:** If the process is interrupted, simply re-run the same command.
> `load_if_exists=True` in `tune.py` picks up where the study left off.

**What this produces:**
```
logs/best_hyperparams.json             ← consumed by Step A-4
logs/optuna_studies/{ALG}_journal.log  ← per-algorithm Optuna log
```

---

## A-4. Step 4 — Tuned Hyperparameters (Local)

Pass `--hyperparams_json` to load the best parameters found in Step A-3.

### Minimal single run

```bash
python train.py \
    --algorithm        CrossQ \
    --set              1 \
    --num_robots       3 \
    --seed             42 \
    --steps            2000000 \
    --experiment       main \
    --hyperparams_json logs/best_hyperparams.json \
    --device           cpu \
    --verbose          1 \
    --log_steps        10000
```

Outputs land in `logs/main_tuned/CrossQ_N3_env1_seed42/`.

### Loop over all algorithms

```bash
for ALG in CrossQ TQC PPO A2C TRPO ARS; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      for N in 2 3 4 5; do
        python train.py \
            --algorithm        $ALG \
            --set              $SET \
            --num_robots       $N \
            --seed             $SEED \
            --steps            2000000 \
            --experiment       main \
            --hyperparams_json logs/best_hyperparams.json \
            --device           cpu \
            --verbose          1 \
            --log_steps        10000
      done
    done
  done
done
```

---

## A-5 to A-7. Ablation Training — Reward, Observation, Uncertainty (Local)

All three ablation experiments use the same `train.py` script with different
`--experiment` and `--ablation` flags.  All fix CrossQ, N=3.

### Step 5 — Reward ablation

```bash
for CONDITION in full no_term no_spr no_path; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      python train.py \
          --algorithm  CrossQ \
          --set        $SET \
          --num_robots 3 \
          --seed       $SEED \
          --steps      2000000 \
          --experiment ablation_reward \
          --ablation   $CONDITION \
          --device     cpu \
          --verbose    1 \
          --log_steps  10000
    done
  done
done
```

### Step 6 — Observation ablation

```bash
for OBS in full no_pos no_inf_hist pos_only; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      python train.py \
          --algorithm  CrossQ \
          --set        $SET \
          --num_robots 3 \
          --seed       $SEED \
          --steps      2000000 \
          --experiment ablation_obs \
          --ablation   $OBS \
          --device     cpu \
          --verbose    1 \
          --log_steps  10000
    done
  done
done
```

### Step 7 — Uncertainty ablation

```bash
for MODE in full wind_only act_only deterministic; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      python train.py \
          --algorithm  CrossQ \
          --set        $SET \
          --num_robots 3 \
          --seed       $SEED \
          --steps      2000000 \
          --experiment ablation_uncertainty \
          --ablation   $MODE \
          --device     cpu \
          --verbose    1 \
          --log_steps  10000
    done
  done
done
```

---

## A-8. Step 8 — Domain Randomization (Local)

```bash
for DR in none wind full; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      for N in 2 3 4 5; do
        python train.py \
            --algorithm  CrossQ \
            --set        $SET \
            --num_robots $N \
            --seed       $SEED \
            --steps      2000000 \
            --experiment dr \
            --ablation   $DR \
            --device     cpu \
            --verbose    1 \
            --log_steps  10000
      done
    done
  done
done
```

---

## A-9. Post-Training Evaluation (Local)

Main results (Tables 1 & 2), transfer-learning results, and observation
ablation (Table 4) are read **directly from training NPZ files** — no separate
evaluation step is needed for those.

The experiments below **do** require `evaluate.py`.

### A-9-A. Reward ablation (Table 3)

```bash
mkdir -p logs/results

for CONDITION in full no_term no_spr no_path; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      python evaluate.py \
          --algorithm  CrossQ \
          --set        $SET \
          --num_robots 3 \
          --seed       $SEED \
          --experiment ablation_reward \
          --ablation   $CONDITION \
          --log_root   logs \
          --output_csv logs/results/ablation_reward.csv \
          --device     cpu
    done
  done
done
```

### A-9-B. Uncertainty ablation cross-evaluation matrix (Table 5)

Each policy trained under one noise condition must be evaluated under all four:

```bash
mkdir -p logs/results

for TRAIN_MODE in full wind_only act_only deterministic; do
  for EVAL_MODE in full wind_only act_only deterministic; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 0 42 123 2024 9999; do
        python evaluate.py \
            --algorithm              CrossQ \
            --set                    $SET \
            --num_robots             3 \
            --seed                   $SEED \
            --experiment             ablation_uncertainty \
            --ablation               $TRAIN_MODE \
            --eval_uncertainty_mode  $EVAL_MODE \
            --log_root               logs \
            --output_csv             logs/results/ablation_uncertainty.csv \
            --device                 cpu
      done
    done
  done
done
```

### A-9-C. Domain randomization — in-distribution & OOD (Table 7)

```bash
mkdir -p logs/results

for DR in none wind full; do
  for SET in 1 2 3 4 5 6 7 8 9 10; do
    for SEED in 0 42 123 2024 9999; do
      for N in 2 3 4 5; do

        # In-distribution: wind 0–0.5 m/s
        python evaluate.py \
            --algorithm     CrossQ \
            --set           $SET \
            --num_robots    $N \
            --seed          $SEED \
            --experiment    dr \
            --ablation      $DR \
            --eval_wind_min 0.0 \
            --eval_wind_max 0.5 \
            --log_root      logs \
            --output_csv    logs/results/dr_inDist.csv \
            --device        cpu

        # Out-of-distribution: wind 0.5–2.0 m/s
        python evaluate.py \
            --algorithm     CrossQ \
            --set           $SET \
            --num_robots    $N \
            --seed          $SEED \
            --experiment    dr \
            --ablation      $DR \
            --eval_wind_min 0.5 \
            --eval_wind_max 2.0 \
            --log_root      logs \
            --output_csv    logs/results/dr_OOD.csv \
            --device        cpu
      done
    done
  done
done
```

### A-9-D. Wind sweep 

```bash
mkdir -p logs/results

wind_mins=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8)
wind_maxs=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

for DR in none full; do
  for BIN in 0 1 2 3 4 5 6 7 8 9; do
    for SEED in 0 42 123 2024 9999; do
      python evaluate.py \
          --algorithm     CrossQ \
          --set           1 \
          --num_robots    3 \
          --seed          $SEED \
          --experiment    dr \
          --ablation      $DR \
          --eval_wind_min ${wind_mins[$BIN]} \
          --eval_wind_max ${wind_maxs[$BIN]} \
          --log_root      logs \
          --output_csv    logs/results/wind_sweep.csv \
          --device        cpu
    done
  done
done
```

---

## A-10. Aggregate Results and Generate Table CSVs (Local)

Identical to the HPC version — no changes needed:

```bash
python analyze_results.py --log_root logs --results_dir logs/results
```

---

## A-11. Generate All Figures (Local)

```bash
python plot_figures.py \
    --log_root    logs \
    --results_dir logs/results \
    --figures_dir figures
```

---

## A-12. Hyperparameter Sensitivity (Table 6, Local)

```bash
for ALG in TRPO CrossQ PPO A2C TQC ARS; do
    python sensitivity_hp.py --algorithm $ALG --results_dir logs/results
done

# Write LaTeX rows after all six finish
python sensitivity_hp.py --write_latex_only --results_dir logs/results
```

---

## Tips for Local Runs

| Goal | Suggestion |
|---|---|
| Smoke-test end-to-end quickly | Use `--steps 50000` and one seed/set |
| Speed up CPU training | Increase `--num_envs` (e.g. `--num_envs 8`) to match your CPU core count |
| Use a GPU | Set `--device cuda`; most helpful for CrossQ and TQC |
| Resume interrupted tuning | Re-run the same `tune.py` command; `load_if_exists=True` continues the study |
| Avoid re-running finished evals | `evaluate.py` automatically skips rows already present in the output CSV |
| Parallelise without SLURM | Run multiple terminal windows or use `&` with `wait` in a shell script |

---

# Part B — HPC Cluster (SLURM)
> All commands assume your working directory is the project root.

Optional: Before submitting jobs on Beocat, create the output/result directories once:

```bash
bash slurm/beocat_prepare_dirs.sh
```

## B-1. Run Step 1 — Default Hyperparameters

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

## B-2. Run Step 2 — Transfer Learning

Step 2 fine-tunes from the Step 1 Set 1 default-policy checkpoint for the
same algorithm, robot count, and seed.  Targets are Sets 2–10.

```bash
sbatch slurm/step2_crossq_transfer.sh     # 180 jobs (1 alg × 9 target sets × 4 N × 5 seeds)
sbatch slurm/step2_others_transfer.sh     # 900 jobs (5 algs × 9 target sets × 4 N × 5 seeds)
```

**Wait for all jobs to finish** before proceeding.

**Source checkpoints:**
```
logs/main_default/{ALG}_N{N}_env1_seed{SEED}/best_model/best_model.zip
```

**What this produces:**
```
logs/main_transfer/{ALG}_N{N}_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
    *.csv  (SB3 training log)
```

---

## B-3. Run Step 3 — Optuna Hyperparameter Tuning

Each SLURM job runs **one Optuna trial**. All jobs for the same algorithm share a
single `JournalStorage` log file (NFS-safe, append-only).

```bash
sbatch slurm/step3_tune_hyperparameters.sh   # 300 jobs (6 algs × 50 trials)
```

**What this produces:**
```
logs/best_hyperparams.json             ← consumed by Step 4
logs/optuna_studies/{ALG}_journal.log  ← per-algorithm Optuna log
```
**Wait for all 300 jobs to finish** before proceeding.

---

## B-4. Run Step 4 — Tuned Hyperparameters

```bash
sbatch slurm/step4_crossq_tuned.sh       # 200 jobs
sbatch slurm/step4_others_tuned.sh       # 1000 jobs
```

**What this produces:**
```
logs/main_tuned/{ALG}_N{N}_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
```

---

## B-5–7. Run Ablation Training — Reward, Observation, Uncertainty

Steps 5, 6, and 7 are handled by a single merged script.
Submit one job array per ablation experiment:

```bash
sbatch --array=0-199 slurm/step5_6_7_ablations.sh ablation_reward       # 200 jobs (4 r_abs × 10 sets × 5 seeds)
sbatch --array=0-199 slurm/step5_6_7_ablations.sh ablation_obs          # 200 jobs (4 obs_abs × 10 sets × 5 seeds)
sbatch --array=0-199 slurm/step5_6_7_ablations.sh ablation_uncertainty  # 200 jobs (4 noise × 10 sets × 5 seeds)
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
> No separate 50-episode evaluation is required for obs ablations.

---

## B-8. Run Step 8 — Domain Randomization

```bash
sbatch slurm/step8_dr.sh           # 600 jobs (3 DR modes × 10 sets × 4 N × 5 seeds)
```

---

## B-9. Post-Training Evaluation

- **Reward ablation (Table 3):** 50-episode rollouts capture terminal-condition
  statistics (`sprayed_pct`, `collision_pct`, `max_steps_pct`).
- **Uncertainty ablation (Table 5):** The cross-evaluation matrix evaluates
  each trained policy under all four noise conditions (not just the one it
  trained on), so 4 train × 4 eval combinations are needed.
- **Domain randomization (Table 7):** Policies must be evaluated under
  controlled in-distribution and out-of-distribution wind ranges.

Main results, transfer learning, tuned results, and obs ablation are filled
directly from the training NPZ files — no separate eval step is required.

> **Re-submission safety:** `evaluate.py` checks whether a matching row already
> exists in the output CSV before running any episodes (matched on algorithm,
> experiment, ablation, hp_tag, num_robots, env_set, seed, eval_wind_min,
> eval_wind_max, and eval_uncertainty_mode).  If found it prints `[SKIP]` and
> exits immediately — so re-submitting failed array indices never produces
> duplicate rows in the CSV.

### B-9-A. Reward ablation (Table 3)

`eval_ablations.sh` runs 50-episode rollouts to capture terminal-condition
statistics (`sprayed_pct`, `collision_pct`, `max_steps_pct`) that the training
NPZ files do not store.  `analyze_results.py` prefers this CSV over the NPZ
fallback when it exists.

> **Note — obs ablation:** Table 4 does **not** need a separate eval step.
> `analyze_results.py` reads the training `evaluations.npz` files directly
> for that experiment.  Skip straight to Step B-10 once Steps B-5–7 training is done.

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

### B-9-B. Uncertainty ablation (Table 5)

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

### B-9-C. Domain randomization (Table 7)

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

### B-9-D. Wind sweep

```bash
sbatch slurm/eval_wind_sweep.sh          # 100 jobs (2 DR × 10 bins × 5 seeds)
```

> `eval_wind_sweep.sh` appends directly to `logs/results/wind_sweep.csv`
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

> **No separate CSVs for main results, transfer learning, tuned results, or obs
> ablations** — those are read from training NPZs by `analyze_results.py`.

Optional: The manual CSV merges above can also be generated with:
```bash
python merge_eval_results.py --results_dir logs/results
```
`analyze_results.py` also attempts these merges before aggregation.

---

## B-9-E. (Optional) Hyperparameter Sensitivity — Table 6

> This step is **independent** of Steps B-1–B-9D and can be run at any time
> after Step B-3 has produced `logs/best_hyperparams.json`.

The sensitivity sweep trains each algorithm over a 7-point grid for each of
its tunable hyperparameters and computes the coefficient of variation (CV) of
IQM across the grid.  Each SLURM job handles one complete algorithm.

```bash
# 6 jobs (one per algorithm) — array=0-5 is declared in the script
sbatch slurm/step9_sensitivity_hp.sh
```

**What this produces:**

```
logs/results/cv_table.csv                  ← one row per (alg, HP)
logs/results/sensitivity_hp_raw.csv        ← one row per grid point
logs/results/sensitivity_hp_latex_rows.txt ← ready-to-paste LaTeX rows for tab:sensitivity_hp
```

After all six jobs finish, regenerate the combined LaTeX table:

```bash
python sensitivity_hp.py --write_latex_only --results_dir logs/results
```

> **Note:** Each algorithm job is independent — if one fails, re-submit only
> that index (e.g., `sbatch --array=2 slurm/step9_sensitivity_hp.sh` for PPO).
> Results are appended to `cv_table.csv` so re-running a failed job is safe.

---

## B-10. Aggregate Results and Generate Table CSVs

Once **all** evaluation jobs are complete:

```bash
python analyze_results.py --log_root logs --results_dir logs/results
```

**What this produces in `logs/results/`:**

| File | Used for | Primary metric |
|---|---|---|
| `main_default_summary.csv` | Tab. 1 data | Mean ± Std reward, IQM |
| `main_transfer_summary.csv` | Transfer-learning data | Mean ± Std reward, IQM |
| `main_tuned_summary.csv`   | Tab. 2 data | Mean ± Std reward, IQM |
| `ablation_reward_agg.csv`  | Tab. 3 data | **Mean episode length ± Std** |
| `ablation_obs_agg.csv`     | Tab. 4 data | IQM of reward, Δ IQM % |
| `ablation_uncertainty_agg.csv` | Tab. 5 data | IQM (same / full / det.) |
| `dr_results_agg.csv`       | Tab. 7 data | In-dist IQM, OOD IQM, CVaR |
| `*_latex_rows.txt`         | **Ready-to-paste LaTeX rows** | — |

> **Data sources:** Main default, transfer-learning, tuned, and observation
> ablation results are read **directly from training `evaluations.npz` files**
> in `--log_root`.  Transfer-learning summaries scan Sets 2–10 only.  Reward
> ablation (Table 3) is read from `logs/results/ablation_reward.csv` (produced
> by `eval_ablations.sh`), with automatic NPZ fallback if the CSV does not
> exist.  Uncertainty ablation (Table 5) and domain randomization (Table 7)
> are read from **evaluation CSVs** in `--results_dir` produced by
> `eval_ablations.sh`.

> **Reward ablation note:** `ablation_reward_agg.csv` and
> `ablation_reward_latex_rows.txt` report **mean episode length** (not IQM of
> reward).  The LaTeX row format is:
> ```
> Condition & Removed terms & Mean reward & Std & Mean ep. length ± Std & Terminal condition \\
> ```
> The condition with the **shortest** mean episode length is bolded (fastest
> task completion).

---

## B-11. Generate All Figures

```bash
python plot_figures.py \
    --log_root    logs \
    --results_dir logs/results \
    --figures_dir figures
```

**Figures produced:**

| File | LaTeX label | Description |
|---|---|---|
| `figures/default_learning_curves.png` | `fig:default_hyp` | Learning curves (default HPs, all N) |
| `figures/default_{N}_robots.png`      | (per-N version)   | Individual N panels |
| `figures/random_learning_curves.png`  | `fig:random_hyp`  | Learning curves (tuned HPs) |
| `figures/random_{N}_robots.png`       | (per-N version)   | Individual N panels |
| `figures/transfer_learning_curves.png` | `fig:transfer` | Learning curves (transfer learning, Sets 2–10) |
| `figures/transfer_{N}_robots.png`     | (per-N version)   | Individual N panels |
| `figures/combined_learning_curves.png` | `fig:combined` | Default + tuned learning curves |
| `figures/scalability.png`             | `fig:scalability` | IQM + ep-length vs N |
| `figures/wind_sensitivity.png`        | `fig:wind_sensitivity` | IQM vs wind speed |
| `figures/dr_curves.png`               | `fig:dr_curves`   | DR training curves |

---

## B-12. Fill LaTeX Tables

### Table 1 & 2 (main results)

1. Open `logs/results/main_default_latex_rows.txt`
2. Each line is a `Algorithm & N=2 & N=3 & N=4 & N=5 \\` row
3. Replace the dummy rows in `full_experiments.tex` between `\midrule` and `\bottomrule` of `tab:default_hyp`
4. Repeat with `main_tuned_latex_rows.txt` for `tab:random_hyp`

Transfer-learning rows are written to `logs/results/main_transfer_latex_rows.txt`
using the same row format.

Example format of each cell:
```
$850.3 \pm 42.1\ (847.2)^\dagger$
```

### Table 3 (reward ablation)

1. Open `logs/results/ablation_reward_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_reward`

Each row now reports **mean episode length ± std** as the primary metric
(shorter = faster task completion = better).  Example cell:
```
Full reward & --- & $910.2$ & $38.4$ & $\mathbf{412.3 \pm 21.1}$ & Sprayed (80\%) \\
```

### Table 4 (observation ablation)

1. Open `logs/results/ablation_obs_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_obs`

### Table 5 (uncertainty ablation)

1. Open `logs/results/ablation_uncertainty_latex_rows.txt`
2. Replace dummy rows in `tab:ablation_uncertainty`

### Table 6 (hyperparameter sensitivity)

This table (CV values) requires a dedicated sensitivity sweep. The values
come from varying each hyperparameter over a 7-point grid.  Run via SLURM
(see **Step B-9-E** above) or manually:

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

1. Open `logs/results/dr_results_latex_rows.txt`
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

## B-13. Update Figure Captions

After generating figures, update the captions and `\includegraphics` paths in
`full_experiments.tex` to point to the correct figure files, e.g.:

```latex
\includegraphics[width=\linewidth]{figures/default_learning_curves.png}
```

---

## Summary of Job Counts

| Step | Script | Jobs |
|---|---|---:|
| 1a | step1_crossq_default.sh | 200 |
| 1b | step1_others_default.sh | 1000 |
| 2a | step2_crossq_transfer.sh | 180 |
| 2b | step2_others_transfer.sh | 900 |
| 3  | step3_tune_hyperparameters.sh | 300 |
| 4a | step4_crossq_tuned.sh   | 200  |
| 4b | step4_others_tuned.sh   | 1000 |
| 5–7a | step5_6_7_ablations.sh (reward)      | 200 |
| 5–7b | step5_6_7_ablations.sh (obs)         | 200 |
| 5–7c | step5_6_7_ablations.sh (uncertainty) | 200 |
| 8  | step8_dr.sh             | 600  |
| 9a | eval_ablations.sh (ablation_reward) | 200 |
| 9b | eval_ablations.sh (uncertainty) — **all 10 sets** | **800** |
| 9c | eval_ablations.sh (dr)          | 600  |
| 9d | eval_wind_sweep.sh      | 100  |
| **Total** | | **6680** |
| 9e *(optional)* | step9_sensitivity_hp.sh | 6 |

> **Step 3:** 300 jobs = 6 algorithms × 50 Optuna trials, one SLURM job per trial
> (the `--array=0-299` is declared inside the script; just `sbatch` the file).
>
> **Compared to the original version:** `eval_main.sh` jobs (2 × 1200 = 2400)
> removed entirely — main results now read from training NPZs.
> Obs ablation eval jobs (200) also removed (reads NPZ directly).
> Uncertainty ablation eval jobs increased from 80 → 800 (all 10 sets).
> Net saving vs original: **~2000 fewer jobs** (excl. optional step 9e).

---

## Dependency Graph (submit in order)

```
Step 1 (default train)
    ├─► analyze_results.py → Tab. 1   (reads NPZ directly — no separate eval)
    │       plot_figures.py → Fig. 1
    │
    └─► Step 2 (transfer learning)
            source: logs/main_default/{ALG}_N{N}_env1_seed{SEED}/best_model/best_model.zip
            target: sets 2–10
            └─► analyze_results.py → transfer summary
                plot_figures.py → transfer curves

Step 3 (tune) ─► Step 4 (tuned train)
                    └─► analyze_results.py → Tab. 2   (reads NPZ directly)
                        plot_figures.py → Fig. 2, 3

Steps 5–7a (step5_6_7_ablations.sh reward)
    └─► eval_ablations.sh (ablation_reward, 200 jobs)
            └─► merge CSVs → logs/results/ablation_reward.csv
                    └─► analyze_results.py → Tab. 3   (CSV preferred; NPZ fallback)

Steps 5–7b (step5_6_7_ablations.sh obs)
    └─► analyze_results.py → Tab. 4   (reads NPZ directly — NO eval step)

Steps 5–7c (step5_6_7_ablations.sh uncertainty)
    └─► eval_ablations.sh (uncertainty, 800 jobs, all 10 sets)
            └─► merge CSVs → logs/results/ablation_uncertainty.csv
                    └─► analyze_results.py → Tab. 5

Step 8 ─► eval_ablations.sh (dr, 600 jobs)
              └─► merge CSVs → logs/results/dr_inDist.csv + logs/results/dr_OOD.csv
                      └─► analyze_results.py → Tab. 7
       └─► eval_wind_sweep.sh → logs/results/wind_sweep.csv
                                    └─► plot_figures.py → Fig. 4, 5

Step 3 ─► (optional) step9_sensitivity_hp.sh (6 jobs)
              └─► python sensitivity_hp.py --write_latex_only → Tab. 6
```


# Simulation  

## CoppeliaSim

First, you will need to download and install the CoppeliaSim robotics simulator from [here](https://coppeliarobotics.com/). Once it is installed, open the `simulation\sim_envs\coppeliasim_scene_for_spraying_v3.ttt` scene file in the simulator. Then follow the instructions given in the jupyter notebook: `simulation\new_env_sim_v3.ipynb`.  

**IMPORTANT:** You will need to reopen the scene each time before running the simulation. Never save changes to the scene file when closing.
