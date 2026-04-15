# EDITED:
import os
import argparse
import traceback
from datetime import datetime

# EDITED:
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, TQC, CrossQ
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from src.utils import load_experiment_dict_json, set_global_seeds

# EDITED:
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'ARS', 'CrossQ', 'PPO', 'TQC', 'TRPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--run_name', type=str, required=True, help='The name of the run, used for logging and saving models')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=500_000, help='The amount of steps to train the DRL model for each Optuna trial')
    parser.add_argument('--num_robots', type=int, default=3, choices=[1, 2, 3, 4, 5, 6, 7], help='Number of robots')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=33, help='The random seed to use')
    parser.add_argument('--n_trials', type=int, default=50, help='The number of Optuna trials to run')
    parser.add_argument('--log_steps', type=int, default=10000, help='The number of steps between each log entry')
    parser.add_argument('--eval_freq', type=int, default=50000, help='The number of steps between each evaluation callback run')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='The number of evaluation episodes to average at the end of each trial')
    parser.add_argument('--device', type=str, default='cpu', help='The device to train on')

    args = parser.parse_args()
    print(args)
    set_global_seeds(args.seed)

    # Configure environment
    json_dict = load_experiment_dict_json('exp_sets/stochastic_envs_v2.json')
    vec_env = make_vec_env('MultiRobotEnv-v0', env_kwargs={'field_info': json_dict[f"set{args.set}"], 'render_mode': None, 'num_robots': args.num_robots}, n_envs=args.num_envs, seed=args.seed) # Make vector environment
    eval_env = make_vec_env('MultiRobotEnv-v0', env_kwargs={'field_info': json_dict[f"set{args.set}"], 'render_mode': None, 'num_robots': args.num_robots}, n_envs=args.num_envs, seed=args.seed) # Make vector environment

    # Logging paths
    log_home = os.path.join('logs', 'training_tuning_logs', f'{args.run_name}')
    os.makedirs(os.path.join(log_home, "logs"), exist_ok=True)
    os.makedirs(os.path.join(log_home, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(log_home, "weights"), exist_ok=True)
    log_path = os.path.join(log_home, 'logs', f"{args.algorithm}_set{args.set}")
    weights_path = os.path.join(log_home, "weights", f"env{args.set}_{args.algorithm}") # trained model weight

    # out_file = os.path.join(log_home, "outputs", f"py_{args.algorithm}_set{args.set}.log") # std output file
    # # Redirect stdout & stderr
    # log_f = open(out_file, "a")
    # sys.stdout = Tee(sys.stdout, log_f)
    # sys.stderr = Tee(sys.stderr, log_f)

    # EDITED:
    mean_rewards = []

    # EDITED:
    def objective(trial):
        logger1 = LogEveryNTimesteps(n_steps=args.log_steps)
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=5, verbose=args.verbose)
        eval_callback = EvalCallback(eval_env, eval_freq=args.eval_freq, callback_after_eval=stop_train_callback, verbose=args.verbose)

        model_args = {
            'env': vec_env,
            'verbose': args.verbose,
            'seed': args.seed,
            'device': args.device,
            'tensorboard_log': log_path + '_optuna',
        }

        if args.algorithm == 'A2C':
            model_args.update({
                'policy': 'MlpPolicy',
                'n_steps': trial.suggest_categorical('n_steps', [5, 10, 20]),
                'gamma': trial.suggest_float('gamma', 0.90, 0.99),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.05),
                'vf_coef': trial.suggest_float('vf_coef', 0.2, 0.7),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 0.99),
            })
            model = A2C(**model_args)
        elif args.algorithm == 'ARS':
            model_args.update({
                'policy': 'LinearPolicy',
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
            })
            model = ARS(**model_args)
        elif args.algorithm == 'CrossQ':
            model_args.update({
                'policy': 'MlpPolicy',
                'gamma': trial.suggest_float('gamma', 0.90, 0.99),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'buffer_size': trial.suggest_int('buffer_size', 1000, 100000, step=1000),
            })
            model = CrossQ(**model_args)
        elif args.algorithm == 'PPO':
            model_args.update({
                'policy': 'MlpPolicy',
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                'gamma': trial.suggest_float('gamma', 0.90, 0.99),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.05),
                'vf_coef': trial.suggest_float('vf_coef', 0.2, 0.7),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 0.99),
            })
            model = PPO(**model_args)
        elif args.algorithm == 'TQC':
            model_args.update({
                'policy': 'MlpPolicy',
                'gamma': trial.suggest_float('gamma', 0.90, 0.99),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
                'buffer_size': trial.suggest_int('buffer_size', 1000, 100000, step=1000),
            })
            model = TQC(**model_args)
        elif args.algorithm == 'TRPO':
            model_args.update({
                'policy': 'MlpPolicy',
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                'gamma': trial.suggest_float('gamma', 0.90, 0.99),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
            })
            model = TRPO(**model_args)
        else:
            raise ValueError('Invalid DRL algorithm!')

        try:
            model.learn(total_timesteps=args.steps, callback=[eval_callback, logger1])
            vec_env.reset()
            eval_env.reset()
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=args.n_eval_episodes, deterministic=True)
        except Exception:
            print('Training failed with error:')
            traceback.print_exc()
            print('*' * 50)
            vec_env.reset()
            eval_env.reset()
            mean_reward = -1e6

        mean_rewards.append(mean_reward)
        return mean_reward

    # EDITED:
    start_time = datetime.now()
    print(f'Tuning started on {start_time.ctime()}')
    print(f'Experiment configuration:', vars(args))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    end_time = datetime.now()
    print(f'Tuning ended on {end_time.ctime()}')
    print(f'Tuning lasted {end_time - start_time}')
    print('Best hyperparameters:', study.best_params)
    print('Best mean reward:', study.best_value)

    scaled_rewards = [x / 1_000_000 for x in mean_rewards]
    x_vals = list(range(len(mean_rewards)))
    plot_path = os.path.join(log_home, 'outputs', f'{args.algorithm}_set{args.set}_optuna_rewards.png')
    plt.figure()
    plt.xticks(x_vals)
    plt.plot(scaled_rewards)
    plt.ylabel('x$10^6$')
    plt.savefig(plot_path)
    plt.close()
    print('Reward plot saved to:', plot_path)
    print('Mean rewards:', mean_rewards)

    # model.save(weights_path)
    vec_env.close()
    eval_env.close()
