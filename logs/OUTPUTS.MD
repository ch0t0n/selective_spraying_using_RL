# Repository Outputs

This file is a consolidated map of the files and folders this repository creates.
It assumes a clean repository where `logs/` starts empty except for this file:

```text
logs/
└── OUTPUTS.MD
```

All paths below are relative to the repository root.

---

## 1. Quick Output Map

| Location | What it contains | Created by |
|---|---|---|
| `logs/main_default/` | Step 1 training runs with default hyperparameters | `train.py`, Step 1 SLURM scripts |
| `logs/main_transfer/` | Step 2 transfer-learning training runs | `train.py`, Step 2 SLURM scripts |
| `logs/optuna_studies/` | Optuna tuning journal logs | `tune.py`, Step 3 SLURM scripts |
| `logs/best_hyperparams.json` | Best hyperparameters found by Optuna | `tune.py` |
| `logs/main_tuned/` | Step 4 training runs using tuned hyperparameters | `train.py`, Step 4 SLURM scripts |
| `logs/ablation_*.../` | Step 5–7 ablation training runs | `train.py`, `step5_6_7_ablations.sh` |
| `logs/dr_*.../` | Step 8 domain-randomization training runs | `train.py`, `step8_dr.sh` |
| `logs/results/tmp/` | Per-job evaluation CSVs before merging | `eval_ablations.sh`, `evaluate.py` |
| `logs/results/` | Merged evaluation CSVs, summary CSVs, and LaTeX row files | `merge_eval_results.py`, `analyze_results.py`, `sensitivity_hp.py` |
| `logs/slurm_outputs/` | SLURM stdout logs | SLURM scripts |
| `logs/slurm_errors/` | SLURM stderr logs | SLURM scripts |
| `figures/` | Paper figures as PNG files | `plot_figures.py` |

---

## 2. Naming Variables Used Below

| Placeholder | Meaning | Values |
|---|---|---|
| `{ALG}` | RL algorithm | `A2C`, `ARS`, `PPO`, `TQC`, `TRPO`, `CrossQ` |
| `{N}` | Number of robots | `2`, `3`, `4`, `5` |
| `{SET}` | Environment set | `1` through `10` |
| `{SEED}` | Random seed | `0`, `42`, `123`, `2024`, `9999` |
| `{CONDITION}` | Reward-ablation condition | `full`, `no_term`, `no_spr`, `no_path` |
| `{OBS_MODE}` | Observation-ablation condition | `full`, `no_pos`, `no_inf_hist`, `pos_only` |
| `{MODE}` | Uncertainty condition | `full`, `wind_only`, `act_only`, `deterministic` |
| `{DR_MODE}` | Domain-randomization condition | `none`, `wind`, `full` |
| `{INDEX}` | SLURM array index | Depends on the job array |

---

## 3. Standard Training Run Folder

Almost every training job writes to one folder with this structure:

```text
logs/{VERSION}/{ALG}_N{N}_env{SET}_seed{SEED}/
├── best_model/
│   └── best_model.zip
├── eval_logs/
│   └── evaluations.npz
├── episode_metrics.csv
├── progress.csv
├── log.txt
├── events.out.tfevents.*
└── {ALG}_N{N}_env{SET}.zip
```

### What each file means

| File | Meaning |
|---|---|
| `best_model/best_model.zip` | Best checkpoint according to the evaluation callback during training. |
| `eval_logs/evaluations.npz` | Evaluation rewards, timesteps, and episode lengths recorded during training. |
| `episode_metrics.csv` | Per-episode environment metrics such as spray use, boundary violations, remaining infection, collision, and success indicators. |
| `progress.csv` | Stable-Baselines3 training metrics in CSV form. |
| `log.txt` | Stable-Baselines3 text log. |
| `events.out.tfevents.*` | TensorBoard event file. |
| `{ALG}_N{N}_env{SET}.zip` | Final model saved after the full training budget finishes. |

---

## 4. Training Output Directories

### Main training outputs

| Experiment step | Directory pattern | Expected run folders |
|---|---:|---:|
| Step 1 — default hyperparameters | `logs/main_default/{ALG}_N{N}_env{SET}_seed{SEED}/` | `1,200` |
| Step 2 — transfer learning | `logs/main_transfer/{ALG}_N{N}_env{SET}_seed{SEED}/` | `1,080` |
| Step 4 — tuned hyperparameters | `logs/main_tuned/{ALG}_N{N}_env{SET}_seed{SEED}/` | `1,200` |

Notes:

- Step 1 covers all six algorithms, ten environment sets, four robot counts, and five seeds.
- Step 2 uses source checkpoints from Set 1 and transfers to Sets 2–10.
- Step 4 uses `logs/best_hyperparams.json` from Step 3.

### Ablation training outputs

| Experiment step | Directory pattern | Expected run folders |
|---|---:|---:|
| Step 5 — reward ablation | `logs/ablation_reward_{CONDITION}/CrossQ_N3_env{SET}_seed{SEED}/` | `200` |
| Step 6 — observation ablation | `logs/ablation_obs_{OBS_MODE}/CrossQ_N3_env{SET}_seed{SEED}/` | `200` |
| Step 7 — uncertainty ablation | `logs/ablation_uncertainty_{MODE}/CrossQ_N3_env{SET}_seed{SEED}/` | `200` |

All ablation training runs use `CrossQ` with `N = 3`.

### Domain-randomization training outputs

| Experiment step | Directory pattern | Expected run folders |
|---|---:|---:|
| Step 8 — domain randomization | `logs/dr_{DR_MODE}/CrossQ_N{N}_env{SET}_seed{SEED}/` | `600` |

---

## 5. Hyperparameter Tuning Outputs

Step 3 tuning creates two types of output.

```text
logs/
├── best_hyperparams.json
└── optuna_studies/
    ├── A2C_journal.log
    ├── ARS_journal.log
    ├── PPO_journal.log
    ├── TRPO_journal.log
    ├── CrossQ_journal.log
    └── TQC_journal.log
```

| File | Meaning |
|---|---|
| `logs/best_hyperparams.json` | Best hyperparameter set found so far for each algorithm. Consumed by Step 4 and by the hyperparameter-sensitivity sweep. |
| `logs/optuna_studies/{ALG}_journal.log` | Append-only Optuna journal used to coordinate tuning trials safely across parallel jobs. |

Step 3 may also create `logs/step3_tune/`. That directory is only a tuning scratch/log-root directory and may be empty.

---

## 6. Post-Training Evaluation Outputs

Some tables are computed directly from training `evaluations.npz` files. Other tables require separate 50-episode evaluation rollouts.

### No separate evaluation CSV is required for these

| Results | Source |
|---|---|
| Main default results | `logs/main_default/.../eval_logs/evaluations.npz` |
| Transfer-learning results | `logs/main_transfer/.../eval_logs/evaluations.npz` |
| Tuned-hyperparameter results | `logs/main_tuned/.../eval_logs/evaluations.npz` |
| Observation ablation | `logs/ablation_obs_*.../eval_logs/evaluations.npz` |

### Evaluation CSV format

When `evaluate.py` writes an evaluation CSV, the columns are:

```text
algorithm, experiment, ablation, hp_tag,
num_robots, env_set, seed,
eval_wind_min, eval_wind_max, eval_uncertainty_mode,
mean_reward, std_reward, max_reward, iqm,
cvar_0.1, mean_ep_length, iqm_ep_length,
sprayed_pct, collision_pct, max_steps_pct,
n_episodes, elapsed_s
```

### Per-job evaluation CSVs

`eval_ablations.sh` writes one small CSV per SLURM array job.

```text
logs/results/tmp/
├── ablation_reward/
│   └── {CONDITION}/set{SET}/result_{INDEX}.csv
├── ablation_uncertainty/
│   └── train_{MODE}_eval_{MODE}/set{SET}/result_{INDEX}.csv
└── dr/
    └── {DR_MODE}/set{SET}_N{N}/
        ├── inDist_{INDEX}.csv
        └── OOD_{INDEX}.csv
```

| Evaluation | Per-job CSV files | Data rows after merge |
|---|---:|---:|
| Reward ablation | `200` | `200` |
| Uncertainty cross-evaluation | `800` | `800` |
| DR in-distribution | `600` | `600` |
| DR out-of-distribution | `600` | `600` |

### Merged evaluation CSVs

`merge_eval_results.py` merges the temporary per-job CSVs into analysis-ready files:

```text
logs/results/
├── ablation_reward.csv
├── ablation_uncertainty.csv
├── dr_inDist.csv
└── dr_OOD.csv
```

### Wind-sweep evaluation CSV

The wind-sweep evaluation appends directly to one file:

```text
logs/results/wind_sweep.csv
```

Expected size: `100` data rows plus one header row.

---

## 7. Analysis Outputs

`analyze_results.py` reads the training NPZ files and merged evaluation CSVs, then writes table-ready outputs to `logs/results/`.

### Main-result summaries

```text
logs/results/
├── main_default_summary.csv
├── main_transfer_summary.csv
├── main_tuned_summary.csv
├── main_default_latex_rows.txt
├── main_transfer_latex_rows.txt
├── main_tuned_latex_rows.txt
└── main_combined_latex_rows.txt
```

| File | Meaning |
|---|---|
| `main_default_summary.csv` | Summary metrics for default-hyperparameter training. |
| `main_transfer_summary.csv` | Summary metrics for transfer-learning training. |
| `main_tuned_summary.csv` | Summary metrics for tuned-hyperparameter training. |
| `main_*_latex_rows.txt` | Ready-to-paste LaTeX table rows for each main-result table. |
| `main_combined_latex_rows.txt` | Compressed IQM-only LaTeX rows comparing default and tuned settings. |

### Ablation and DR summaries

```text
logs/results/
├── ablation_reward_agg.csv
├── ablation_obs_agg.csv
├── ablation_uncertainty_agg.csv
├── dr_results_agg.csv
├── ablation_reward_latex_rows.txt
├── ablation_obs_latex_rows.txt
├── ablation_uncertainty_latex_rows.txt
├── dr_results_latex_rows.txt
└── combined_ablations_dr_latex_rows.txt
```

| File | Meaning |
|---|---|
| `ablation_reward_agg.csv` | Aggregated reward-ablation metrics, including terminal-condition summaries when `ablation_reward.csv` exists. |
| `ablation_obs_agg.csv` | Aggregated observation-ablation metrics. |
| `ablation_uncertainty_agg.csv` | Aggregated uncertainty cross-evaluation metrics. |
| `dr_results_agg.csv` | Aggregated in-distribution, OOD, and CVaR metrics for domain randomization. |
| `*_latex_rows.txt` | Ready-to-paste LaTeX rows for the corresponding paper or supplement tables. |
| `combined_ablations_dr_latex_rows.txt` | Combined LaTeX rows for ablation and DR results. |

---

## 8. Figure Outputs

`plot_figures.py` writes PNG files to `figures/`.

```text
figures/
├── default_learning_curves.png
├── default_2_robots.png
├── default_3_robots.png
├── default_4_robots.png
├── default_5_robots.png
├── random_learning_curves.png
├── random_2_robots.png
├── random_3_robots.png
├── random_4_robots.png
├── random_5_robots.png
├── transfer_learning_curves.png
├── transfer_2_robots.png
├── transfer_3_robots.png
├── transfer_4_robots.png
├── transfer_5_robots.png
├── combined_learning_curves.png
├── scalability.png
├── wind_sensitivity.png
└── dr_curves.png
```

| Figure | Inputs needed |
|---|---|
| `default_*` | `logs/main_default/.../eval_logs/evaluations.npz` |
| `random_*` | `logs/main_tuned/.../eval_logs/evaluations.npz` |
| `transfer_*` | `logs/main_transfer/.../eval_logs/evaluations.npz` |
| `combined_learning_curves.png` | Default and tuned training NPZ files. |
| `scalability.png` | `logs/results/main_tuned_summary.csv`. |
| `wind_sensitivity.png` | `logs/results/wind_sweep.csv`. |
| `dr_curves.png` | `logs/dr_none/`, `logs/dr_wind/`, and `logs/dr_full/` training NPZ files. |

If a required input is missing, the corresponding plot may be skipped or generated without the missing curves.

---

## 9. Hyperparameter-Sensitivity Outputs

`sensitivity_hp.py` writes all hyperparameter-sensitivity results to `logs/results/`.

```text
logs/results/
├── sensitivity_hp_raw.csv
├── cv_table.csv
└── sensitivity_hp_latex_rows.txt
```

| File | Meaning | Expected data rows when complete |
|---|---|---:|
| `sensitivity_hp_raw.csv` | One row per algorithm, hyperparameter, and grid value. | `189` |
| `cv_table.csv` | One row per algorithm and hyperparameter, with coefficient of variation. | `27` |
| `sensitivity_hp_latex_rows.txt` | Ready-to-paste LaTeX rows for the sensitivity table. | Not row-count based |

The raw row count is `27` hyperparameter sweeps × `7` grid values each.

---

## 10. SLURM Output and Error Logs

When running on an HPC cluster, SLURM writes job stdout and stderr files.

```text
logs/slurm_outputs/
├── s1_crossq_default/
├── s1_others_default/
├── s2_crossq_transfer/
├── s2_others_transfer/
├── s3_tune/
├── s3_crossq_tune/
├── s3_others_tune/
├── s4_crossq_tuned/
├── s4_others_tuned/
├── s5_6_7_ablations/
├── s8_dr/
├── eval_ablations/
├── eval_wind_sweep/
└── step9_sens/
```

```text
logs/slurm_errors/
├── s1_crossq_default/
├── s1_others_default/
├── s2_crossq_transfer/
├── s2_others_transfer/
├── s3_tune/
├── s3_crossq_tune/
├── s3_others_tune/
├── s4_crossq_tuned/
├── s4_others_tuned/
├── s5_6_7_ablations/
├── s8_dr/
└── step9_sens/
```

Notes:

- Most SLURM stdout files use names like `{JOB_NAME}_{SLURM_JOB_ID}_{ARRAY_ID}.out`.
- Most SLURM stderr files use names like `{JOB_NAME}_{SLURM_JOB_ID}_{ARRAY_ID}.err`.
- `eval_ablations.sh` sends both `.out` and `.err` files into `logs/slurm_outputs/eval_ablations/`.
- `eval_wind_sweep.sh` sends both `.out` and `.err` files into `logs/slurm_outputs/eval_wind_sweep/`.
- `check_gpu.sh` writes `logs/gpu_check_{SLURM_JOB_ID}.log`.

---

## 11. Directory Setup Outputs

`beocat_prepare_dirs.sh` creates output directories before jobs are submitted.
It creates directories only; it does not create result data.

Expected directories include:

```text
logs/
logs/optuna_studies/
logs/results/tmp/
logs/slurm_outputs/.../
logs/slurm_errors/.../
figures/
```

---

## 12. Status-Checker Outputs

These scripts are reporting tools. They print readiness/status information to the terminal and do not create result files.

```text
check_status_train_tune.py
check_status_ablation_dr.py
check_status_all_train_eval.py
```

Use them to verify whether expected training, tuning, evaluation, analysis, and plotting outputs exist.

---

## 13. Final Expected High-Level Tree

After a full HPC run, merge, analysis, plotting, and sensitivity sweep, the repository should contain a tree similar to this:

```text
logs/
├── OUTPUTS.MD
├── best_hyperparams.json
├── optuna_studies/
│   └── {ALG}_journal.log
├── main_default/
│   └── {ALG}_N{N}_env{SET}_seed{SEED}/...
├── main_transfer/
│   └── {ALG}_N{N}_env{SET}_seed{SEED}/...
├── main_tuned/
│   └── {ALG}_N{N}_env{SET}_seed{SEED}/...
├── ablation_reward_{CONDITION}/
│   └── CrossQ_N3_env{SET}_seed{SEED}/...
├── ablation_obs_{OBS_MODE}/
│   └── CrossQ_N3_env{SET}_seed{SEED}/...
├── ablation_uncertainty_{MODE}/
│   └── CrossQ_N3_env{SET}_seed{SEED}/...
├── dr_{DR_MODE}/
│   └── CrossQ_N{N}_env{SET}_seed{SEED}/...
├── results/
│   ├── tmp/
│   ├── ablation_reward.csv
│   ├── ablation_uncertainty.csv
│   ├── dr_inDist.csv
│   ├── dr_OOD.csv
│   ├── wind_sweep.csv
│   ├── main_default_summary.csv
│   ├── main_transfer_summary.csv
│   ├── main_tuned_summary.csv
│   ├── ablation_reward_agg.csv
│   ├── ablation_obs_agg.csv
│   ├── ablation_uncertainty_agg.csv
│   ├── dr_results_agg.csv
│   ├── cv_table.csv
│   ├── sensitivity_hp_raw.csv
│   └── *_latex_rows.txt
├── slurm_outputs/
└── slurm_errors/

figures/
└── *.png
```

---

## 14. Minimal Completeness Checklist

A full reproduction is broadly complete when these outputs exist:

- `logs/best_hyperparams.json`
- Six files in `logs/optuna_studies/`: one `{ALG}_journal.log` per algorithm
- Training run folders for Steps 1, 2, 4, 5, 6, 7, and 8
- `logs/results/ablation_reward.csv`
- `logs/results/ablation_uncertainty.csv`
- `logs/results/dr_inDist.csv`
- `logs/results/dr_OOD.csv`
- `logs/results/wind_sweep.csv`
- `logs/results/main_default_summary.csv`
- `logs/results/main_transfer_summary.csv`
- `logs/results/main_tuned_summary.csv`
- `logs/results/ablation_reward_agg.csv`
- `logs/results/ablation_obs_agg.csv`
- `logs/results/ablation_uncertainty_agg.csv`
- `logs/results/dr_results_agg.csv`
- `logs/results/cv_table.csv`
- `logs/results/sensitivity_hp_raw.csv`
- `logs/results/sensitivity_hp_latex_rows.txt`
- All expected `logs/results/*_latex_rows.txt` files
- All expected `figures/*.png` files

The status-checker scripts can be used to confirm whether the files above are present and whether important run folders contain valid model checkpoints and `evaluations.npz` files.
