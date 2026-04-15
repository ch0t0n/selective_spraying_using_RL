# EDITED:
import ast
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# EDITED:
TRIAL_RESULT_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+finished\s+with\s+value:\s*"
    r"(?P<value>[-+0-9.eEinfINFnanNAN]+)\s+and\s+parameters:\s*"
    r"(?P<params>\{.*\})"
)
RUN_NAME_RE = re.compile(r"RUN_NAME=(?P<run_name>\S+)")
ALG_LINE_RE = re.compile(
    r"ALG=(?P<algorithm>\S+)\s+SET=(?P<set>\d+)\s+SEED=(?P<seed>-?\d+)\s+DEVICE=(?P<device>\S+)"
)
RUN_INFO_RE = re.compile(
    r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)_seed(?P<seed>-?\d+)_"
    r"(?P<exp_name>[^_]+)_(?P<num_robots>\d+)_robots_(?P<device>cpu|cuda)$"
)


# EDITED:
num_robots = 3
exp_name = 'random'
date = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

os.makedirs('plotting/results', exist_ok=True)
os.makedirs('plotting/plots', exist_ok=True)


# EDITED:
TRIAL_STATE_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+(?P<state>finished|pruned|failed)\b",
    re.IGNORECASE,
)
VALUE_RE = re.compile(r"value:\s*(?P<value>[-+0-9.eEinfINFnanNAN]+)")
PARAMS_RE = re.compile(r"parameters:\s*(?P<params>\{.*\})")


# EDITED:
def read_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except FileNotFoundError:
        return ''


# EDITED:
def parse_trial_line(line):
    trial_state_match = TRIAL_STATE_RE.search(line)
    if trial_state_match is None:
        return None
    if trial_state_match.group('state').lower() != 'finished':
        return None

    value_match = VALUE_RE.search(line)
    if value_match is None:
        return None

    try:
        reward = float(value_match.group('value'))
    except ValueError:
        return None
    if not np.isfinite(reward):
        return None

    params = None
    params_match = PARAMS_RE.search(line)
    if params_match is not None:
        params = params_match.group('params')
        try:
            params = ast.literal_eval(params)
        except Exception:
            pass

    return {
        'trial': int(trial_state_match.group('trial')),
        'reward': reward,
        'params': params,
    }


# # EDITED:
# data = []
# log_path = os.path.join('slurm_scripts', 'slurm_out', '*.out')
# logs = glob.glob(log_path)
# for log in logs:
#     with open(log, 'r', encoding='utf-8', errors='replace') as f:
#         text = f.read()
#
#     run_name_match = RUN_NAME_RE.search(text)
#     run_name = run_name_match.group('run_name') if run_name_match else None
#     if not run_name or f"_{exp_name}_{num_robots}_robots_" not in run_name:
#         continue
#
#     algorithm = None
#     st = None
#     seed = None
#     device = None
#
#     alg_line_match = ALG_LINE_RE.search(text)
#     if alg_line_match is not None:
#         algorithm = alg_line_match.group('algorithm')
#         st = f"set{alg_line_match.group('set')}"
#         seed = int(alg_line_match.group('seed'))
#         device = alg_line_match.group('device')
#     else:
#         run_info_match = RUN_INFO_RE.match(run_name)
#         if run_info_match is None:
#             continue
#         algorithm = run_info_match.group('algorithm')
#         st = f"set{run_info_match.group('set')}"
#         seed = int(run_info_match.group('seed'))
#         device = run_info_match.group('device')
#
#     log_mtime = os.path.getmtime(log)
#     for line in text.splitlines():
#         trial_match = TRIAL_RESULT_RE.search(line)
#         if trial_match is None:
#             continue
#
#         try:
#             reward = float(trial_match.group('value'))
#         except ValueError:
#             continue
#         if not np.isfinite(reward):
#             continue
#
#         params = trial_match.group('params')
#         try:
#             params = ast.literal_eval(params)
#         except Exception:
#             pass
#
#         data.append({
#             'algorithm': algorithm,
#             'set': st,
#             'trial': int(trial_match.group('trial')),
#             'reward': reward,
#             'run_name': run_name,
#             'seed': seed,
#             'device': device,
#             'params': params,
#             'slurm_file': log,
#             'log_mtime': log_mtime,
#         })
#
#
# df = pd.DataFrame(data)
# if df.empty:
#     raise ValueError(
#         "No Optuna tuning trial results were found in slurm_scripts/slurm_out. "
#         "Expected Optuna 'Trial ... finished with value ...' lines in Slurm .out files."
#     )

# EDITED:
data = []
log_path = os.path.join('slurm_scripts', 'slurm_out', '*.out')
logs = glob.glob(log_path)
for log in logs:
    out_text = read_text(log)
    err_log = os.path.splitext(log)[0] + '.err'
    err_text = read_text(err_log)
    combined_text = "\n".join(part for part in [out_text, err_text] if part)

    run_name_match = RUN_NAME_RE.search(out_text) or RUN_NAME_RE.search(combined_text)
    run_name = run_name_match.group('run_name') if run_name_match else None
    if not run_name or f"_{exp_name}_{num_robots}_robots_" not in run_name:
        continue

    algorithm = None
    st = None
    seed = None
    device = None

    alg_line_match = ALG_LINE_RE.search(out_text) or ALG_LINE_RE.search(combined_text)
    if alg_line_match is not None:
        algorithm = alg_line_match.group('algorithm')
        st = f"set{alg_line_match.group('set')}"
        seed = int(alg_line_match.group('seed'))
        device = alg_line_match.group('device')
    else:
        run_info_match = RUN_INFO_RE.match(run_name)
        if run_info_match is None:
            continue
        algorithm = run_info_match.group('algorithm')
        st = f"set{run_info_match.group('set')}"
        seed = int(run_info_match.group('seed'))
        device = run_info_match.group('device')

    mtime_candidates = [os.path.getmtime(log)]
    if os.path.exists(err_log):
        mtime_candidates.append(os.path.getmtime(err_log))
    log_mtime = max(mtime_candidates)

    for line in combined_text.splitlines():
        parsed = parse_trial_line(line)
        if parsed is None:
            continue

        data.append({
            'algorithm': algorithm,
            'set': st,
            'trial': parsed['trial'],
            'reward': parsed['reward'],
            'run_name': run_name,
            'seed': seed,
            'device': device,
            'params': parsed['params'],
            'slurm_file': log,
            'slurm_err_file': err_log if os.path.exists(err_log) else None,
            'log_mtime': log_mtime,
        })


df = pd.DataFrame(data)
if df.empty:
    raise ValueError(
        "No Optuna tuning trial results were found in slurm_scripts/slurm_out. "
        "Expected Optuna trial lines in the paired Slurm .out/.err files."
    )


# EDITED:
df = df.sort_values(['run_name', 'trial', 'log_mtime']).drop_duplicates(
    subset=['run_name', 'trial'],
    keep='last',
)
df['params'] = df['params'].astype(str)
df.to_csv(f"plotting/results/wheeled_{exp_name}_tuning_raw_{date}.csv", index=False)


# EDITED:
df['reward_scaled'] = df['reward'] / 1000000
df['reward_scaled'] = df['reward_scaled'].clip(lower=-2)

min_step = df['trial'].min()
max_step = df['trial'].max()
common_steps = np.linspace(min_step, max_step, 200)

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10, 6))

for algo, algo_df in df.groupby('algorithm'):
    runs = []

    for run_id, run_df in algo_df.groupby('set'):
        x = run_df['trial'].values
        y = run_df['reward_scaled'].values

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_unique, unique_idx = np.unique(x, return_index=True)
        y_unique = y[unique_idx]

        if len(x_unique) == 1:
            interp_rewards = np.full_like(common_steps, y_unique[0], dtype=float)
        else:
            interp_rewards = np.interp(common_steps, x_unique, y_unique)
        runs.append(interp_rewards)

    runs = np.array(runs)
    mean_rewards = runs.mean(axis=0)
    std_rewards = runs.std(axis=0)

    plt.plot(common_steps, mean_rewards, label=algo, linewidth=2)
    plt.fill_between(common_steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

plt.xlabel("Trial")
plt.ylabel("Reward (x$10^6$)")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plotting/plots/original_{exp_name}_tuning_graph_for_{num_robots}_robots.png")
plt.show()


# EDITED:
A2C_df = df[df['algorithm'] == 'A2C']
ARS_df = df[df['algorithm'] == 'ARS']
CrossQ_df = df[df['algorithm'] == 'CrossQ']
PPO_df = df[df['algorithm'] == 'PPO']
TQC_df = df[df['algorithm'] == 'TQC']
TRPO_df = df[df['algorithm'] == 'TRPO']

A2C_data = {
    "Setting": "Tuning",
    "Algorithm": "A2C",
    "Mean": A2C_df['reward'].mean(),
    "Max": A2C_df['reward'].max(),
    "SD": A2C_df['reward'].std(),
    "Range": (A2C_df['reward'].max() - A2C_df['reward'].min()),
}
ARS_data = {
    "Setting": "Tuning",
    "Algorithm": "ARS",
    "Mean": ARS_df['reward'].mean(),
    "Max": ARS_df['reward'].max(),
    "SD": ARS_df['reward'].std(),
    "Range": (ARS_df['reward'].max() - ARS_df['reward'].min()),
}
CrossQ_data = {
    "Setting": "Tuning",
    "Algorithm": "CrossQ",
    "Mean": CrossQ_df['reward'].mean(),
    "Max": CrossQ_df['reward'].max(),
    "SD": CrossQ_df['reward'].std(),
    "Range": (CrossQ_df['reward'].max() - CrossQ_df['reward'].min()),
}
PPO_data = {
    "Setting": "Tuning",
    "Algorithm": "PPO",
    "Mean": PPO_df['reward'].mean(),
    "Max": PPO_df['reward'].max(),
    "SD": PPO_df['reward'].std(),
    "Range": (PPO_df['reward'].max() - PPO_df['reward'].min()),
}
TQC_data = {
    "Setting": "Tuning",
    "Algorithm": "TQC",
    "Mean": TQC_df['reward'].mean(),
    "Max": TQC_df['reward'].max(),
    "SD": TQC_df['reward'].std(),
    "Range": (TQC_df['reward'].max() - TQC_df['reward'].min()),
}
TRPO_data = {
    "Setting": "Tuning",
    "Algorithm": "TRPO",
    "Mean": TRPO_df['reward'].mean(),
    "Max": TRPO_df['reward'].max(),
    "SD": TRPO_df['reward'].std(),
    "Range": (TRPO_df['reward'].max() - TRPO_df['reward'].min()),
}

set_tab_df = pd.DataFrame([A2C_data, ARS_data, CrossQ_data, PPO_data, TQC_data, TRPO_data])
set_tab_df_sc = set_tab_df.select_dtypes(include='number') / 1000000
set_tab_df_sc.insert(0, 'Setting', set_tab_df['Setting'])
set_tab_df_sc.insert(1, 'Algorithm', set_tab_df['Algorithm'])

set_tab_df_sc_trans = set_tab_df_sc.T
set_tab_df_sc_trans.to_csv(f'plotting/results/original_{exp_name}_tuning_results_for_{num_robots}_robots.csv')
