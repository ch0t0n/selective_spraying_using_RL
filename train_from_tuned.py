# EDITED:
import os
import sys
import ast
import re
import argparse
from pathlib import Path
from datetime import datetime
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, TQC, CrossQ
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure
from src.utils import load_experiment_dict_json, load_model, parse_bool, Tee, set_global_seeds

# EDITED:
RUN_NAME_RE = re.compile(r"RUN_NAME=(?P<run_name>\S+)")
BEST_PARAMS_RE = re.compile(r"Best hyperparameters:\s*(?P<params>\{.*\})")


# EDITED:
def read_text(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''


# EDITED:
def parse_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == '':
        return None
    try:
        return int(value)
    except ValueError:
        return None


# EDITED:
def safe_literal_eval(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


# EDITED:
def infer_tuning_run_name(args):
    if args.tuning_run_name is not None:
        return args.tuning_run_name
    return f"{args.algorithm}_set{args.set}_seed{args.seed}_{args.tuning_exp_name}_{args.num_robots}_robots_{args.device}"


# EDITED:
def load_tuned_hyperparameters(tuning_run_name, slurm_out_dir):
    slurm_out_dir = Path(slurm_out_dir)
    if not slurm_out_dir.exists():
        raise FileNotFoundError(f"Slurm output directory does not exist: {slurm_out_dir}")

    pair_map = {}
    for path in sorted(slurm_out_dir.iterdir()):
        if not path.is_file() or path.suffix not in {'.out', '.err'}:
            continue

        parts = path.stem.rsplit('_', 2)
        if len(parts) != 3:
            continue
        job_name, job_id, task_text = parts
        task_id = parse_int(task_text)
        if task_id is None:
            continue

        key = (job_name, job_id, task_id)
        pair = pair_map.get(key)
        if pair is None:
            pair = {
                'job_name': job_name,
                'job_id': job_id,
                'task_id': task_id,
                'out_path': None,
                'err_path': None,
                'out_text': '',
                'err_text': '',
                'mtime_rank': -1,
            }
            pair_map[key] = pair

        if path.suffix == '.out':
            pair['out_path'] = path
            pair['out_text'] = read_text(path)
        else:
            pair['err_path'] = path
            pair['err_text'] = read_text(path)

        try:
            pair['mtime_rank'] = max(pair['mtime_rank'], int(path.stat().st_mtime))
        except OSError:
            pass

    best_match = None
    for pair in pair_map.values():
        combined_text = '\n'.join(part for part in [pair['out_text'], pair['err_text']] if part)
        run_name_match = RUN_NAME_RE.search(combined_text)
        if run_name_match is None:
            continue
        if run_name_match.group('run_name') != tuning_run_name:
            continue

        params_matches = list(BEST_PARAMS_RE.finditer(combined_text))
        if not params_matches:
            continue

        job_id_rank = int(pair['job_id']) if str(pair['job_id']).isdigit() else -1
        rank = (job_id_rank, pair['mtime_rank'])
        if best_match is None or rank >= best_match['rank']:
            best_match = {
                'rank': rank,
                'raw_params': params_matches[-1].group('params'),
                'out_path': pair['out_path'],
                'err_path': pair['err_path'],
            }

    if best_match is None:
        raise FileNotFoundError(
            'Could not find tuned hyperparameters for '
            f'{tuning_run_name} under {slurm_out_dir}. '
            'The tuning flow currently persists them in the Slurm stdout/stderr '
            'via the "Best hyperparameters:" line.'
        )

    tuned_hyperparameters = safe_literal_eval(best_match['raw_params'])
    if not isinstance(tuned_hyperparameters, dict):
        raise ValueError(
            'Best hyperparameters were found, but they could not be parsed into a dict '
            f'for tuning run {tuning_run_name}.'
        )

    source_paths = {
        'out_path': str(best_match['out_path']) if best_match['out_path'] is not None else None,
        'err_path': str(best_match['err_path']) if best_match['err_path'] is not None else None,
    }
    return tuned_hyperparameters, source_paths


# EDITED:
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'ARS', 'CrossQ', 'PPO', 'TQC', 'TRPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--run_name', type=str, required=True, help='The name of the run, used for logging and saving models. If not provided, a name will be generated based on the algorithm, set, and seed')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=200_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_robots', type=int, default=3, choices=[1, 2, 3, 4, 5, 6, 7], help='Number of robots')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=10000, help='The number of steps between each log entry')
    parser.add_argument('--resume', type=parse_bool, default=False, help='If true, loads an existing model to resume training. If false, trains a new model')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to train on')
    # EDITED:
    parser.add_argument('--tuning_exp_name', type=str, default='random', help='Experiment suffix used by the tuning run name when locating tuned hyperparameters')
    parser.add_argument('--tuning_run_name', type=str, default=None, help='Optional explicit tuning run name to load best hyperparameters from')
    parser.add_argument('--tuning_slurm_out_dir', type=str, default=os.path.join('slurm_scripts', 'slurm_out'), help='Directory containing the Slurm stdout/stderr from tuning runs')

    args = parser.parse_args()
    print(args)
    set_global_seeds(args.seed)

    # EDITED:
    tuning_run_name = infer_tuning_run_name(args)
    tuned_hyperparameters, tuned_hyperparameter_sources = load_tuned_hyperparameters(
        tuning_run_name=tuning_run_name,
        slurm_out_dir=args.tuning_slurm_out_dir,
    )
    print(f'Loaded tuned hyperparameters from {tuning_run_name}:', tuned_hyperparameters)
    print('Tuned hyperparameter sources:', tuned_hyperparameter_sources)

    # Configure environment
    json_dict = load_experiment_dict_json(f'exp_sets/stochastic_envs_v2.json')
    vec_env = make_vec_env('MultiRobotEnv-v0', env_kwargs={'field_info':json_dict[f"set{args.set}"], 'render_mode': None, 'num_robots':args.num_robots}, n_envs = args.num_envs, seed=args.seed) # Make vector environment

    # Logging paths
    # EDITED:
    log_home = os.path.join('logs', 'training_from_tuned_logs', f'{args.run_name}')
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

    # Set loggers
    logger1 = LogEveryNTimesteps(n_steps=args.log_steps)
    logger2 = configure(log_path, ["stdout", "csv", "tensorboard"])

    # Configure model
    if args.resume:
        model = load_model(args.algorithm, args.set, args.seed, args.device, 'trained_models', args.verbose, 'training_logs')
        model.set_env(vec_env)
    else:
        model_args = {
            'policy': 'MlpLstmPolicy' if args.algorithm == 'RecurrentPPO' else 'MlpPolicy',
            'env': vec_env,
            'verbose': args.verbose,
            # 'tensorboard_log': log_home+f'_{args.algorithm}',
            'seed': args.seed,
            'device': args.device,
        }

        # EDITED:
        model_args.update(tuned_hyperparameters)

        if args.algorithm == 'A2C':
            model = A2C(**model_args)
        elif args.algorithm == 'PPO':
            model = PPO(**model_args)
        elif args.algorithm == 'TRPO':
            model = TRPO(**model_args)
        elif args.algorithm == 'ARS':
            model = ARS(**model_args)
        elif args.algorithm == 'CrossQ':
            model = CrossQ(**model_args)
        elif args.algorithm == 'TQC':
            model = TQC(**model_args)
        else:
            raise ValueError("Invalid DRL algorithm!")

    # Train model
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()}')
    print(f'Experiment configuration:', vars(args))
    # EDITED:
    print('Tuned hyperparameters used for training:', tuned_hyperparameters)
    model.set_logger(logger2)
    model.learn(total_timesteps=args.steps, callback=logger1, log_interval=None, tb_log_name=f"{args.algorithm}_set{args.set}", reset_num_timesteps=False)
    end_time = datetime.now()
    print(f'Training ended on {end_time.ctime()}')
    print(f'Training lasted {end_time - start_time}')

    # Save model
    model.save(weights_path)
    vec_env.close()
