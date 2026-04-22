import json
import os
# import yaml
import numpy as np
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, CrossQ, TQC
import distutils
import inspect

# Loads in a trained model
def load_model(algorithm, st, num_robots=3, seed=42, log_root="logs",
               version="main_default", model_path=None, device="cpu"):
    if model_path is None:
        active_path = os.path.join(
            log_root,
            version,
            f"{algorithm}_N{num_robots}_env{st}_seed{seed}",
            "best_model",
            "best_model.zip",
        )
        legacy_path = f'trained_models/{algorithm}_set{st}.zip'
        model_path = active_path if os.path.exists(active_path) else legacy_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if algorithm == 'A2C':
        model = A2C.load(model_path, device=device)
    elif algorithm == 'PPO':
        model = PPO.load(model_path, device=device)
    elif algorithm == 'TRPO':
        model = TRPO.load(model_path, device=device)
    elif algorithm == 'ARS':
        model = ARS.load(model_path, device=device)
    elif algorithm == 'CrossQ':
        model = CrossQ.load(model_path, device=device)
    elif algorithm == 'TQC':
        model = TQC.load(model_path, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
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
    action = int(action)
    x = action % 5
    y = (action // 5) % 5
    z = action // 25
    return np.array([x, y, z])

# Filters out arguments that are not present in a model's constructor
def filter_args(args, model):
    model_kwargs = inspect.getfullargspec(model).args
    return {k:args[k] for k in args if k in model_kwargs}

# Function to check if a point is inside a polygon (Ray-casting algorithm)
def is_inside_polygon(point, poly):  # check if a point is inside polygon
    x, y = point  # unpack point
    inside = False  # initialize flag
    n = len(poly)  # number of vertices
    p1x, p1y = poly[0]  # first vertex
    for i in range(n + 1):  # iterate through edges
        p2x, p2y = poly[i % n]  # next vertex (wrap around)
        # check if point is within vertical bounds of edge
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:  # avoid division by zero
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x  # intersection
            if p1x == p2x or x <= xinters:  # crossing check
                inside = not inside  # toggle
        p1x, p1y = p2x, p2y  # move to next edge
    return inside  # return result

# Function to return minimum distance in a list of points
def compute_min_dist(x):  # compute minimum pairwise distance
    x = np.asarray(x, dtype=np.float32)  # ensure array
    diff = x[:, None, :] - x[None, :, :]  # pairwise differences
    dist_matrix = np.linalg.norm(diff, axis=-1)  # compute distances
    np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance
    return float(np.min(dist_matrix))  # return min distance

# Load experiment json file
def load_experiment_dict_json(json_path):  # load experiment configs
    with open(json_path, "r") as f:  # open file
        data = json.load(f)  # load JSON
    for _, cfg in data.items():  # iterate over configs
        cfg["field"] = [tuple(p) for p in cfg["field"]]  # convert to tuples
        cfg["init_positions"] = np.array(cfg["init_positions"], dtype=float)  # positions array
        cfg["infected_locations"] = [tuple(p) for p in cfg["infected_locations"]]  # infection
    return data  # return processed dict

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


def set_global_seeds(seed: int) -> None:
    if seed is None:
        return

    import os
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass