import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Gather data

num_robots = 3
exp_name = 'default'
date = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

data = []
log_path = os.path.join('logs', f'training_{exp_name}_logs', f'*_{exp_name}_{num_robots}_robots_*', 'logs', '*', 'events.out.tfevents*') # Set the path for the experiments
logs = glob.glob(log_path)
for log in logs:
    if "events.out.tfevents" in log:
        experiment_info = log.split(os.sep)[-2].split('_')
        algorithm = experiment_info[0]
        st = experiment_info[1]
        # print(log)
        # print(log.split(os.sep))
        # print(experiment_info)
        
        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    else:
        # print(f"Not a tensorflow log file, file: {log}")
        raise ValueError(f"Not a tensorflow log file, file: {log}")
df = pd.DataFrame(data)
df.to_csv(f"plotting/results/wheeled_default_raw_{date}.csv")


# Scaling the rewards for visualization
df['reward_scaled'] = df['reward'] / 1000000
df['reward_scaled'] = df['reward_scaled'].clip(lower=-2)

# Define a common x-axis (based on min-max steps across all algorithms)
min_step = df['step'].min()
max_step = df['step'].max()
common_steps = np.linspace(min_step, max_step, 200)

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,6))

# Group by algorithm
for algo, algo_df in df.groupby('algorithm'):
    runs = []

    # For each run (set), interpolate rewards to common steps
    for run_id, run_df in algo_df.groupby('set'):
        x = run_df['step'].values
        y = run_df['reward_scaled'].values
        interp_rewards = np.interp(common_steps, x, y)
        runs.append(interp_rewards)

    runs = np.array(runs)
    mean_rewards = runs.mean(axis=0)
    std_rewards = runs.std(axis=0)

    # Plot with variance shading
    plt.plot(common_steps, mean_rewards, label=algo, linewidth=2)
    plt.fill_between(common_steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

plt.xlabel("Step")
plt.ylabel("Reward (x$10^6$)")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plotting/plots/original_{exp_name}_graph_for_{num_robots}_robots.png")
plt.show()

A2C_df = df[df['algorithm']=='A2C']
ARS_df = df[df['algorithm']=='ARS']
CrossQ_df = df[df['algorithm']=='CrossQ']
PPO_df = df[df['algorithm']=='PPO']
TQC_df = df[df['algorithm']=='TQC']
TRPO_df = df[df['algorithm']=='TRPO']


A2C_data = {"Setting": "AB", 
            "Algorithm": "A2C", 
            "Mean": A2C_df['reward'].mean(),
            "Max": A2C_df['reward'].max(),
            "SD": A2C_df['reward'].std(),
            "Range": (A2C_df['reward'].max() - A2C_df['reward'].min())}
ARS_data = {"Setting": "AB", 
            "Algorithm": "ARS", 
            "Mean": ARS_df['reward'].mean(),            
            "Max": ARS_df['reward'].max(),
            "SD": ARS_df['reward'].std(),
            "Range": (ARS_df['reward'].max() - ARS_df['reward'].min())}
CrossQ_data = {"Setting": "AB", 
            "Algorithm": "CrossQ", 
            "Mean": CrossQ_df['reward'].mean(),            
            "Max": CrossQ_df['reward'].max(),
            "SD": CrossQ_df['reward'].std(),
            "Range": (CrossQ_df['reward'].max() - CrossQ_df['reward'].min())}
PPO_data = {"Setting": "AB", 
            "Algorithm": "PPO", 
            "Mean": PPO_df['reward'].mean(),
            "Max": PPO_df['reward'].max(),
            "SD": PPO_df['reward'].std(),            
            "Range": (PPO_df['reward'].max() - PPO_df['reward'].min())}
TQC_data = {"Setting": "AB", 
            "Algorithm": "TQC", 
            "Mean": TQC_df['reward'].mean(),            
            "Max": TQC_df['reward'].max(),
            "SD": TQC_df['reward'].std(),
            "Range": (TQC_df['reward'].max() - TQC_df['reward'].min())}
TRPO_data = {"Setting": "AB", 
            "Algorithm": "TRPO", 
            "Mean": TRPO_df['reward'].mean(),            
            "Max": TRPO_df['reward'].max(),
            "SD": TRPO_df['reward'].std(),
            "Range": (TRPO_df['reward'].max() - TRPO_df['reward'].min())}


set_tab_df = pd.DataFrame([A2C_data, ARS_data, CrossQ_data, PPO_data, TQC_data, TRPO_data])


set_tab_df_sc = set_tab_df.select_dtypes(include='number') / 1000000
set_tab_df_sc.insert(0, 'Setting', set_tab_df['Setting'])
set_tab_df_sc.insert(1, 'Algorithm', set_tab_df['Algorithm'])


set_tab_df_sc_trans = set_tab_df_sc.T
set_tab_df_sc_trans.to_csv(f'plotting/results/original_{exp_name}_results_for_{num_robots}_robots.csv')
