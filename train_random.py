import os
import sys
import argparse
from datetime import datetime
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, TQC, CrossQ
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure
from src.utils import load_experiment_dict_json, load_model, parse_bool, Tee

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'ARS', 'CrossQ', 'PPO', 'TQC', 'TRPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=1_000_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_robots', type=int, choices=[1, 2, 3, 4, 5, 6, 7], default=3, help='Number of robots')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=10000, help='The number of steps between each log entry')
    parser.add_argument('--resume', type=parse_bool, default=False, help='If true, loads an existing model to resume training. If false, trains a new model')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to train on')
    args = parser.parse_args()
    print(args)
    
    # Configure environment
    if args.num_robots > 3:
        json_dict = load_experiment_dict_json(f'experiments/new_2026_sets_five.json')
    else:
        json_dict = load_experiment_dict_json(f'experiments/new_2026_cont_sets.json')
    vec_env = make_vec_env('MultiRobotEnv-v0', env_kwargs={'field_info':json_dict[f"set{args.set}"], 'render_mode': None, 'num_robots':args.num_robots}, n_envs = args.num_envs, seed=args.seed) # Make vector environment
    
    # Logging paths
    log_home = os.path.join('training_random_logs', f'{args.num_robots}_robots')
    os.makedirs(os.path.join(log_home, "logs"), exist_ok=True)
    os.makedirs(os.path.join(log_home, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(log_home, "weights"), exist_ok=True)
    log_path = os.path.join(log_home, 'logs', f"{args.algorithm}_set{args.set}")
    out_file = os.path.join(log_home, "outputs", f"py_{args.algorithm}_set{args.set}.log") # std output file
    weights_path = os.path.join(log_home, "weights", f"env{args.set}_{args.algorithm}") # trained model weight
    
    # Redirect stdout & stderr
    log_f = open(out_file, "a")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)  

    # Set loggers
    logger1 = LogEveryNTimesteps(n_steps=args.log_steps)
    logger2 = configure(log_path, ["stdout", "log", "csv", "json", "tensorboard"])

    # Configure model
    if args.resume:
        model = load_model(args.algorithm, args.set, args.seed, args.device, 'trained_models', args.verbose, 'training_logs')
        model.set_env(vec_env)
    else:
        if args.algorithm == 'A2C':
            model = A2C("MlpPolicy", 
                        vec_env,
                        seed=args.seed,
                        verbose=1,
                        device=args.device,
                        tensorboard_log=log_path+f'_{args.algorithm}',
                        n_steps=20,
                        gamma=0.9042,
                        learning_rate=0.0001,
                        ent_coef=0.0470,
                        vf_coef=0.570065,
                        gae_lambda=0.9660,
                        max_grad_norm=0.7739)
        elif args.algorithm == 'PPO':
            model = PPO("MlpPolicy", 
                        vec_env, 
                        seed=args.seed,
                        verbose=1,
                        device=args.device,
                        tensorboard_log=log_path+f'_{args.algorithm}',
                        n_steps=2048,
                        gamma=0.9411,
                        learning_rate=0.0017,
                        ent_coef=0.0477,
                        vf_coef=0.5042,
                        gae_lambda=0.9837,
                        max_grad_norm=0.4559)
        elif args.algorithm == 'TRPO':
            model = TRPO("MlpPolicy", 
                        vec_env, 
                        seed=args.seed,
                        verbose=1,
                        device=args.device,
                        tensorboard_log=log_path+f'_{args.algorithm}',
                        n_steps=2048,
                        gamma=0.9606,
                        learning_rate=0.0013,
                        gae_lambda=0.9811)
        elif args.algorithm == 'ARS':
            model = ARS("MlpPolicy", 
                        vec_env,
                        seed=args.seed,
                        verbose=1,
                        device=args.device,
                        tensorboard_log=log_path+f'_{args.algorithm}',
                        learning_rate=0.0018)
        elif args.algorithm == 'CrossQ':
            model = CrossQ("MlpPolicy", 
                        vec_env, 
                        seed=args.seed,
                        verbose=1,
                        device=args.device,
                        tensorboard_log=log_path+f"_{args.algorithm}",
                        gamma=0.9425,
                        learning_rate=0.0055,
                        buffer_size=34000)
        elif args.algorithm == 'TQC':
            model = TQC("MlpPolicy", 
                        vec_env, 
                        seed=args.seed,
                        verbose=1,
                        device=args.device,
                        tensorboard_log=log_path+f"_{args.algorithm}",
                        gamma=0.9363,
                        learning_rate=0.0064,
                        buffer_size=69000)
        else:
            raise ValueError("Invalid DRL algorithm!")

    # Train model
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()}')
    print(f'Experiment configuration:', vars(args))
    model.set_logger(logger2)
    model.learn(total_timesteps=args.steps, callback=logger1, log_interval=None, tb_log_name=f"{args.algorithm}_set{args.set}", reset_num_timesteps=False)
    end_time = datetime.now()
    print(f'Training ended on {end_time.ctime()}')
    print(f'Training lasted {end_time - start_time}')
    
    # Save model
    model.save(weights_path)
    vec_env.close()
