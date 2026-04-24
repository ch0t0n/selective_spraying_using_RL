# Repository outputs and logging overview

This file summarizes every major log, generated output, output location, and naming convention used by the repository.

## Path variables

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `PROJECT_ROOT` | inferred from the script location | Python scripts and SLURM scripts | Repository root. |
| `LOG_ROOT` | `$PROJECT_ROOT/logs` | Training, tuning, evaluation scripts | Root directory for model checkpoints, training logs, Optuna files, and scheduler logs. |
| `RESULTS_DIR` | `$PROJECT_ROOT/results` | Evaluation SLURM scripts and analysis | Root directory for raw and aggregate result CSVs. |
| `BEST_JSON` | `$LOG_ROOT/best_hyperparams.json` | Tuning and tuned-training scripts | Shared tuned-hyperparameter JSON. |
| `JOURNAL_DIR` | `$LOG_ROOT/optuna_studies` | Tuning scripts | Directory for Optuna journal files. |

The Python defaults now resolve through `PROJECT_ROOT`, so invoking `train.py`, `tune.py`, `evaluate.py`, or `analyze_results.py` from outside the repository root still targets the repository’s `exp_sets/`, `logs/`, and `results/` directories.

## Top-level output map

```text
logs/
  main_default/
  main_transfer/
  main_tuned/
  ablation_reward_<condition>/
  ablation_obs_<condition>/
  ablation_uncertainty_<condition>/
  dr_<mode>/
    {algorithm}_N{num_robots}_env{set}_seed{seed}/episode_metrics.csv
  step3_tune/
  optuna_studies/
  slurm_outputs/
  slurm_errors/
  best_hyperparams.json
  best_hyperparams.json.lock

results/
  results_default.csv
  results_tuned.csv
  results_transfer.csv
  ablation_reward.csv
  ablation_obs.csv
  ablation_uncertainty.csv
  dr_inDist.csv
  dr_OOD.csv
  wind_sweep.csv
  *_summary.csv
  *_agg.csv
  *_latex_rows.txt
```

## Training outputs

Training is run through `train.py`. The main output directory is:

```text
$LOG_ROOT/{version}/{algorithm}_N{num_robots}_env{set}_seed{seed}/
```

`version` is determined by the experiment type:

| Experiment | Version directory |
|---|---|
| Main, default hyperparameters | `main_default` |
| Main, transfer from Set 1 default checkpoint | `main_transfer` |
| Main, tuned hyperparameters | `main_tuned` |
| Reward ablation | `ablation_reward_<condition>` |
| Observation ablation | `ablation_obs_<condition>` |
| Uncertainty ablation | `ablation_uncertainty_<condition>` |
| Domain randomization | `dr_<mode>` |

Each training run may generate:

| Output | Location | Description |
|---|---|---|
| Best model checkpoint | `best_model/best_model.zip` | Saved by `EvalCallback` when evaluation performance improves. |
| Evaluation callback log | `eval_logs/evaluations.npz` | Stable-Baselines evaluation callback data. |
| Training episode metrics | `episode_metrics.csv` | One row per completed training episode, captured from terminal `info["episode_metrics"]`. |
| Final model checkpoint | `{algorithm}_N{num_robots}_env{set}.zip` | Saved after `model.learn(...)` completes. |
| SB3 logger text/log output | run directory | Stable-Baselines logger output from the `log` backend. |
| SB3 logger CSV output | run directory, commonly `progress.csv` | Training progress metrics from the `csv` backend. |
| TensorBoard event files | run directory | TensorBoard logs from the `tensorboard` backend. |
| Episode metric aggregates | `progress.csv` and TensorBoard | Training-time means logged by the episode-metrics callback. |
| Console output | SLURM stdout file or terminal | Run header, hyperparameter loading messages, training progress, and final save path. |

The run directory includes the seed to prevent parallel jobs with the same algorithm, robot count, and environment set from overwriting each other.

Transfer-learning runs load source checkpoints from:

```text
$LOG_ROOT/main_default/{algorithm}_N{num_robots}_env1_seed{seed}/best_model/best_model.zip
```

and write target fine-tuning runs for Sets 2–10 under `main_transfer`.

## Hyperparameter tuning outputs

Tuning is run through `tune.py` and the tuning SLURM scripts.

| Output | Location | Description |
|---|---|---|
| Optuna journal | `$LOG_ROOT/optuna_studies/{algorithm}_journal.log` | Shared append-only Optuna study log. |
| Best hyperparameters | `$LOG_ROOT/best_hyperparams.json` | JSON file containing best IQM score, parameter dictionary, and tuning context per algorithm. |
| Best-hyperparameter lock | `$LOG_ROOT/best_hyperparams.json.lock` | File lock used while updating the shared JSON. |
| Temporary best-HP file | `$LOG_ROOT/best_hyperparams.json.tmp.<pid>` | Atomic-write temporary file used before replacement. |
| Tuning log root | `$LOG_ROOT/step3_tune` | Tuning log directory passed to `tune.py`. |
| Console output | SLURM stdout file or terminal | Algorithm/device/storage/study header, trial IQM lines, failures, best-so-far summary, and update message. |

The best-hyperparameter JSON uses this structure:

```json
{
  "AlgorithmName": {
    "iqm": 123.45,
    "params": {
      "learning_rate": 0.001
    },
    "context": {
      "set": 1,
      "num_robots": 3,
      "tune_seed": 42,
      "tune_steps": 500000
    }
  }
}
```

## Evaluation outputs

Evaluation is run through `evaluate.py` and writes one upserted row per evaluated run. The row is inserted or replaced under a file lock so repeated evaluations of the same key do not append duplicates.

For main-experiment evaluations, `hp_tag` may be `default`, `tuned`, or `transfer`. Transfer evaluations reuse the same main-result CSV schema and write to `results/results_transfer.csv`. Under the current scripts, transfer rows record `pretrain_steps=2000000`, `finetune_steps=2000000`, and `total_train_steps=4000000`.

For DR evaluations, `ablation` records the training/checkpoint DR mode, while `eval_dr_mode` records the environment DR mode used for evaluation.

### Raw result CSVs

| CSV | Produced by | Description |
|---|---|---|
| `results/results_default.csv` | `eval_main.sh default` | Main experiment evaluations for default hyperparameters. |
| `results/results_tuned.csv` | `eval_main.sh tuned` | Main experiment evaluations for tuned hyperparameters. |
| `results/results_transfer.csv` | `eval_main.sh transfer` | Main experiment evaluations for transfer-learned policies on Sets 2–10. |
| `results/ablation_reward.csv` | `eval_ablations.sh ablation_reward` | Reward-ablation evaluations. |
| `results/ablation_obs.csv` | `eval_ablations.sh ablation_obs` | Observation-ablation evaluations. |
| `results/ablation_uncertainty.csv` | `eval_ablations.sh ablation_uncertainty` | Train uncertainty mode × eval uncertainty mode matrix. |
| `results/dr_inDist.csv` | `eval_ablations.sh dr` | Domain-randomization evaluations with wind in `[0.0, 1.0]`. |
| `results/dr_OOD.csv` | `eval_ablations.sh dr` | Domain-randomization evaluations with wind in `[1.0, 2.0]`. |
| `results/wind_sweep.csv` | `eval_wind_sweep.sh` | Wind-bin sweep for standard vs full-DR CrossQ policies. |

Every raw CSV receives a sibling lock file while being updated:

```text
<csv_path>.lock
```

### Evaluation CSV schema

`evaluate.py` writes these columns:

```text
algorithm
experiment
ablation
hp_tag
num_robots
env_set
seed
pretrain_steps
finetune_steps
total_train_steps
hparam_search_steps_per_algorithm
total_train_steps_plus_search
eval_wind_min
eval_wind_max
eval_uncertainty_mode
eval_reward_ablation
eval_dr_mode
mean_reward
std_reward
max_reward
iqm
cvar_0.1
mean_ep_length
n_episodes
elapsed_s
success_rate
collision_rate
time_limit_rate
mean_episode_spray_used
mean_episode_spray_wasted
mean_episode_spray_applied
mean_episode_wasted_fraction
mean_episode_spray_empty_capacity
mean_episode_boundary_viol_count
mean_episode_remaining_infection_sum
mean_episode_remaining_infection_fraction
episode_rewards_json
episode_lengths_json
episode_metrics_json
```

The upsert key is:

```text
algorithm, experiment, ablation, hp_tag,
num_robots, env_set, seed,
eval_wind_min, eval_wind_max,
eval_uncertainty_mode, eval_reward_ablation, eval_dr_mode
```

### Evaluation console output

Each evaluation prints:

- the model path being loaded;
- algorithm, experiment, ablation, environment set, robot count, seed, and episode count;
- mean reward, standard deviation, max reward, IQM, CVaR 0.1, mean episode length, elapsed seconds, success rate, collision rate, and remaining-infection fraction;
- the CSV path that was updated.

## Analysis outputs

Analysis is run through `analyze_results.py`. It consumes raw CSVs from `results/` and writes aggregate CSVs and LaTeX snippets.

| Output | Description |
|---|---|
| `results/main_default_summary.csv` | Aggregated main results for default hyperparameters. |
| `results/main_default_latex_rows.txt` | LaTeX rows for the default-hyperparameter main table. |
| `results/main_tuned_summary.csv` | Aggregated main results for tuned hyperparameters. |
| `results/main_tuned_latex_rows.txt` | LaTeX rows for the tuned-hyperparameter main table. |
| `results/main_transfer_summary.csv` | Aggregated main results for transfer-learned policies. |
| `results/main_transfer_latex_rows.txt` | LaTeX rows for the transfer-learning main table. |
| `results/ablation_reward_agg.csv` | Aggregated reward-ablation results. |
| `results/ablation_reward_latex_rows.txt` | LaTeX rows for reward ablation. |
| `results/ablation_obs_agg.csv` | Aggregated observation-ablation results. |
| `results/ablation_obs_latex_rows.txt` | LaTeX rows for observation ablation. |
| `results/ablation_uncertainty_agg.csv` | Aggregated uncertainty cross-evaluation matrix. |
| `results/ablation_uncertainty_latex_rows.txt` | LaTeX rows for uncertainty ablation. |
| `results/dr_results_agg.csv` | Aggregated domain-randomization results. |
| `results/dr_results_latex_rows.txt` | LaTeX rows for domain randomization. |

The analysis script prints warnings when optional input CSVs are missing, writes confirmation lines for each generated file, prints compact main-result summary tables, and ends by reporting the LaTeX row-file location.

## Figure outputs

Figure generation is run through `plot_figures.py`. It consumes training logs from `logs/` and summary/evaluation CSVs from `results/`.

| Output | Description |
|---|---|
| `figures/default_learning_curves.png` | Combined default-hyperparameter learning curves. |
| `figures/default_{N}_robots.png` | Per-robot-count default learning curves. |
| `figures/random_learning_curves.png` | Combined tuned-hyperparameter learning curves. |
| `figures/random_{N}_robots.png` | Per-robot-count tuned learning curves. |
| `figures/transfer_learning_curves.png` | Combined transfer fine-tuning learning curves for target Sets 2–10. |
| `figures/transfer_{N}_robots.png` | Per-robot-count transfer fine-tuning curves. |
| `figures/scalability.png` | Tuned-policy IQM and episode length versus number of robots. |
| `figures/transfer_scalability.png` | Transfer-policy IQM and episode length versus number of robots. |
| `figures/wind_sensitivity.png` | IQM versus wind speed. |
| `figures/dr_curves.png` | Domain-randomization training curves. |

## SLURM scheduler outputs

All SLURM stdout and stderr paths now use the same repository-relative convention:

```text
logs/slurm_outputs/<job-name>/...
logs/slurm_errors/<job-name>/...
```

| Script | Stdout | Stderr |
|---|---|---|
| `check_gpu.sh` | `logs/slurm_outputs/gpu_check/%x_%j.out` | `logs/slurm_errors/gpu_check/%x_%j.err` |
| `step1_crossq_default.sh` | `logs/slurm_outputs/s1_crossq_default/%x_%A_%a.out` | `logs/slurm_errors/s1_crossq_default/%x_%A_%a.err` |
| `step1_others_default.sh` | `logs/slurm_outputs/s1_others_default/%x_%A_%a.out` | `logs/slurm_errors/s1_others_default/%x_%A_%a.err` |
| `step2_crossq_transfer.sh` | `logs/slurm_outputs/s2_crossq_transfer/%x_%A_%a.out` | `logs/slurm_errors/s2_crossq_transfer/%x_%A_%a.err` |
| `step2_others_transfer.sh` | `logs/slurm_outputs/s2_others_transfer/%x_%A_%a.out` | `logs/slurm_errors/s2_others_transfer/%x_%A_%a.err` |
| `step3_tune_cpu.sh` | `logs/slurm_outputs/s3_tune_cpu/%x_%A_%a.out` | `logs/slurm_errors/s3_tune_cpu/%x_%A_%a.err` |
| `step3_tune_gpu.sh` | `logs/slurm_outputs/s3_tune_gpu/%x_%A_%a.out` | `logs/slurm_errors/s3_tune_gpu/%x_%A_%a.err` |
| `step4_crossq_tuned.sh` | `logs/slurm_outputs/s4_crossq_tuned/%x_%A_%a.out` | `logs/slurm_errors/s4_crossq_tuned/%x_%A_%a.err` |
| `step4_others_tuned.sh` | `logs/slurm_outputs/s4_others_tuned/%x_%A_%a.out` | `logs/slurm_errors/s4_others_tuned/%x_%A_%a.err` |
| `step5_ablation_reward.sh` | `logs/slurm_outputs/s5_ablation_reward/%x_%A_%a.out` | `logs/slurm_errors/s5_ablation_reward/%x_%A_%a.err` |
| `step6_ablation_obs.sh` | `logs/slurm_outputs/s6_ablation_obs/%x_%A_%a.out` | `logs/slurm_errors/s6_ablation_obs/%x_%A_%a.err` |
| `step7_ablation_uncertainty.sh` | `logs/slurm_outputs/s7_ablation_uncertainty/%x_%A_%a.out` | `logs/slurm_errors/s7_ablation_uncertainty/%x_%A_%a.err` |
| `step8_dr.sh` | `logs/slurm_outputs/s8_dr/%x_%A_%a.out` | `logs/slurm_errors/s8_dr/%x_%A_%a.err` |
| `eval_main.sh` | `logs/slurm_outputs/eval_main/%x_%A_%a.out` | `logs/slurm_errors/eval_main/%x_%A_%a.err` |
| `eval_ablations.sh` | `logs/slurm_outputs/eval_ablation/%x_%A_%a.out` | `logs/slurm_errors/eval_ablation/%x_%A_%a.err` |
| `eval_wind_sweep.sh` | `logs/slurm_outputs/eval_wind_sweep/%x_%A_%a.out` | `logs/slurm_errors/eval_wind_sweep/%x_%A_%a.err` |

`%x` is the SLURM job name, `%j` is the job ID, `%A` is the array job ID, and `%a` is the array task ID.

## Environment runtime outputs

`MultiRobotEnv.step()` returns an `info` dictionary containing these step-level fields:

```text
total_sprayed
remaining_infection
episode_length
path_length
```

At episode end (`terminated` or `truncated`), the same `info` dictionary also contains:

```text
episode_metrics
```

`episode_metrics` is a dictionary with these terminal episode fields:

```text
episode_step_count
episode_spray_used
episode_spray_wasted
episode_spray_applied
episode_wasted_fraction
episode_spray_empty_capacity
episode_boundary_viol_count
episode_remaining_infection_sum
episode_remaining_infection_fraction
episode_time_limit_reached
episode_collision_occurred
episode_success
```

`episode_spray_empty_capacity` records spray commands that could not be emitted because capacity was empty or insufficient.

These values are not written directly to disk by the environment. `train.py` saves training episode metrics in each run directory, and `evaluate.py` saves final-evaluation episode metrics in the raw result CSVs.

`MultiRobotEnv.render()` opens a Pygame human-rendering window and draws the field, robot trajectories, robots, and infected locations. It does not save image files.

## Generated-output conventions

- Model directories are keyed by experiment version, algorithm, robot count, environment set, and seed.
- Evaluation CSV rows are keyed by the run identity plus evaluation-specific overrides such as wind range, uncertainty mode, and reward-ablation mode.
- Raw evaluation CSVs are safe for repeated runs because `evaluate.py` uses a lock and replaces matching keys.
- Analysis outputs are deterministic aggregations of the raw CSV inputs available in `results/`.
- SLURM output paths are repository-relative because SLURM parses `#SBATCH` directives before shell variables are assigned.
