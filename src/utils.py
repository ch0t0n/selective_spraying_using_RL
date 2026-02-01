import json
import itertools
import yaml
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO, ARS, RecurrentPPO
import distutils
import inspect

# Loads in an experiment config file
def load_experiment(path):
    with open(path, 'r') as experiment_file:
        config = yaml.load(experiment_file, Loader=yaml.FullLoader)
        config['field'] = list(map(lambda x: tuple(x), config['field']))
        config['init_positions'] = list(map(lambda x: np.array(x), config['init_positions']))
        config['infected_locations'] = list(map(lambda x: tuple(x), config['infected_locations']))
    return config

# Loads in a trained model
def load_model(algorithm, experiment_set, seed, device, models_dir, verbose, log_dir):
    model_args = {
        'path': f'{models_dir}/{algorithm}_set{experiment_set}.zip',
        'tb_log_name': f'{algorithm}_set{experiment_set}',
        'device': device,
        'seed': seed,
        'verbose': verbose,
        'tensorboard_log': log_dir,
    }

    if algorithm == 'A2C':
        model = A2C.load(**model_args)
    elif algorithm == 'PPO':
        model = PPO.load(**model_args)
    elif algorithm == 'TRPO':
        model = TRPO.load(**model_args)
    elif algorithm == 'DQN':
        model = DQN.load(**model_args)
    elif algorithm == 'ARS':
        model = ARS.load(**model_args)
    else:
        model = RecurrentPPO.load(**model_args)
    return model

# Converts a list of binary digits to its decimal equivalent
def binary_list_to_decimal(bin_list):
    bin = ''
    for b in bin_list:
        bin += str(b)
    dec = int(bin,2)
    return dec

# Parses a string into a bool
def parse_bool(string):
    return bool(distutils.util.strtobool(string))

# Encoding function: (x, y, z) → Discrete
def encode_action(action):
    x, y, z = action
    return x + 5 * y + 25 * z  # 25 = 5 * 5

# Decoding function: Discrete → (x, y, z)
def decode_action(action):
    x = action // 25
    y = (action // 5) % 5
    z = action % 5
    return np.array([x, y, z])

# Filters out arguments that are not present in a model's constructor
def filter_args(args, model):
    model_kwargs = inspect.getfullargspec(model).args
    return {k:args[k] for k in args if k in model_kwargs}

# Function to check if a point is inside a polygon (Ray-casting algorithm)
def is_inside_polygon(point, poly):
    x, y = point
    inside = False
    n = len(poly)
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Function to return minimum distance in a list of points
def compute_min_dist(x):
    x = np.array(x).astype('float32')
    dists = []
    for p1, p2 in itertools.combinations(x, 2):
        dist = np.linalg.norm(p1-p2)
        dists.append(dist)
    return float(np.min(dists))

# Load experiment json file
def load_experiment_dict_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    for set_name, cfg in data.items():
        # Convert field to list of tuples
        cfg["field"] = [tuple(p) for p in cfg["field"]]
        # Convert init_positions to NumPy arrays
        cfg["init_positions"] = [np.array(p, dtype=float) for p in cfg["init_positions"]]
        # Convert infected_locations to set of tuples
        cfg["infected_locations"] = [tuple(p) for p in cfg["infected_locations"]]
    return data

# =========================
# STDOUT / STDERR REDIRECT
# =========================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
