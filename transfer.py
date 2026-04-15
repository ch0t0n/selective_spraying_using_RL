# EDITED:
import os
import sys
import argparse
from datetime import datetime
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, TQC, CrossQ
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure
from src.utils import load_experiment_dict_json, parse_bool, Tee, set_global_seeds


# EDITED:
def build_source_run_name(algorithm, load_set, seed, num_robots, device):
    return f"{algorithm}_set{load_set}_seed{seed}_default_{num_robots}_robots_{device}"


# EDITED:
def load_transfer_model(algorithm, source_weights_path, device, seed, verbose):
    load_args = {
        'path': source_weights_path,
        'device': device,
    }

    if algorithm == 'A2C':
        model = A2C.load(**load_args)
    elif algorithm == 'PPO':
        model = PPO.load(**load_args)
    elif algorithm == 'TRPO':
        model = TRPO.load(**load_args)
    elif algorithm == 'ARS':
        model = ARS.load(**load_args)
    elif algorithm == 'CrossQ':
        model = CrossQ.load(**load_args)
    elif algorithm == 'TQC':
        model = TQC.load(**load_args)
    else:
        raise ValueError("Invalid DRL algorithm!")

    # EDITED:
    try:
        model.verbose = verbose
    except Exception:
        pass

    # EDITED:
    try:
        model.set_random_seed(seed)
    except Exception:
        pass

    return model


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'ARS', 'CrossQ', 'PPO', 'TQC', 'TRPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The target experiment set to continue training on')
    parser.add_argument('--run_name', type=str, required=True, help='The name of the run, used for logging and saving models')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=200_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_robots', type=int, default=3, choices=[1, 2, 3, 4, 5, 6, 7], help='Number of robots')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=10000, help='The number of steps between each log entry')
    # EDITED:
    parser.add_argument('--load_set', type=int, default=1, help='The source set whose trained model will be loaded before transfer training')
    # EDITED:
    parser.add_argument('--source_run_name', type=str, default=None, help='Optional explicit source run name. If omitted, the standard default-training run name is inferred.')
    parser.add_argument('--resume', type=parse_bool, default=False, help='Unused for transfer runs. Present only to keep argument parity with train_default.py')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to train on')

    args = parser.parse_args()
    print(args)
    set_global_seeds(args.seed)

    # Configure environment
    json_dict = load_experiment_dict_json('exp_sets/stochastic_envs_v2.json')
    vec_env = make_vec_env('MultiRobotEnv-v0', env_kwargs={'field_info': json_dict[f"set{args.set}"], 'render_mode': None, 'num_robots': args.num_robots}, n_envs=args.num_envs, seed=args.seed) # Make vector environment

    # Logging paths
    # EDITED:
    log_home = os.path.join('logs', 'training_transfer_logs', f'{args.run_name}')
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

    # Configure source model path
    # EDITED:
    source_run_name = args.source_run_name or build_source_run_name(
        args.algorithm,
        args.load_set,
        args.seed,
        args.num_robots,
        args.device,
    )
    # EDITED:
    source_log_home = os.path.join('logs', 'training_default_logs', source_run_name)
    # EDITED:
    source_weights_path = os.path.join(source_log_home, "weights", f"env{args.load_set}_{args.algorithm}.zip")

    # EDITED:
    if not os.path.exists(source_weights_path):
        raise FileNotFoundError(
            "Could not find the source pretrained model for transfer learning. "
            f"Expected: {source_weights_path}"
        )

    # Configure model
    # EDITED:
    model = load_transfer_model(
        args.algorithm,
        source_weights_path,
        args.device,
        args.seed,
        args.verbose,
    )
    # EDITED:
    model.set_env(vec_env)

    # Train model
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()}')
    print(f'Experiment configuration:', vars(args))
    # EDITED:
    print(f'Source run name: {source_run_name}')
    # EDITED:
    print(f'Source weights path: {source_weights_path}')
    model.set_logger(logger2)
    model.learn(total_timesteps=args.steps, callback=logger1, log_interval=None, tb_log_name=f"{args.algorithm}_set{args.set}", reset_num_timesteps=False)
    end_time = datetime.now()
    print(f'Training ended on {end_time.ctime()}')
    print(f'Training lasted {end_time - start_time}')

    # Save model
    model.save(weights_path)
    vec_env.close()
