import os  # OS operations (paths, folders)
import json  # JSON file handling
import datetime  # timestamping logs
import numpy as np  # numerical operations
import gymnasium as gym  # RL environment framework
import pygame  # rendering

from stable_baselines3.common.callbacks import LogEveryNTimesteps, EvalCallback, CallbackList  # logging callback
from stable_baselines3.common.env_util import make_vec_env  # vectorized env
from stable_baselines3.common.logger import configure  # logger
from sb3_contrib import CrossQ  # RL algorithm

# ================================
# Utility Functions (Optimized)
# ================================

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

def compute_min_dist(x):  # compute minimum pairwise distance
    x = np.asarray(x, dtype=np.float32)  # ensure array
    diff = x[:, None, :] - x[None, :, :]  # pairwise differences
    dist_matrix = np.linalg.norm(diff, axis=-1)  # compute distances
    np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance
    return float(np.min(dist_matrix))  # return min distance

def load_experiment_dict_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    for _, cfg in data.items():
        cfg["field"] = [tuple(p) for p in cfg["field"]]
        cfg["init_positions"] = np.array(cfg["init_positions"], dtype=float)
        cfg["infected_locations"] = [tuple(p) for p in cfg["infected_locations"]]
    return data


# ================================
# Environment
# ================================

class MultiRobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 60}
    def __init__(self, field_info, render_mode=None, wind_par=[0, 0], num_robots=3, render_scale=10, max_steps=1000):
        super().__init__()

        # Screen / Field
        self.field_info = field_info
        self.render_scale = render_scale # scaling for rendering
        self.poly_vertices = field_info['field']
        self.xs, self.ys = zip(*self.poly_vertices) # separate coordinates
        self.WIDTH, self.HEIGHT = max(self.xs) + 10, max(self.ys) + 10 # screen size
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.max_steps = max_steps # max episode length

        # Robot params
        self.num_robots = num_robots
        self.robot_size = 2 # collision radius
        self.mass = 1.0 # robot mass
        self.thrust_power = 0.5 # action scaling
        self.max_speed = 5
        self.min_speed = -5        

        self.init_robot_positions = field_info['init_positions'][:num_robots]
        self.init_robot_capacities = np.array(field_info['robot_capacities'][:num_robots], dtype=np.float32)

        # Infection (VECTOR FORM)
        infected = field_info['infected_locations']
        self.infected_positions_init = np.array([[x, y] for x, y, _ in infected])
        self.infected_levels_init = np.array([lvl for _, _, lvl in infected], dtype=np.float32)
        self.max_infection_level = np.max(self.infected_levels_init)
        self.infected_radius = 5 # spray radius
        self.spray_sigma = self.infected_radius / 2 # Gaussian spread

        # Base wind (mean wind)
        self.base_wind_mag, self.base_wind_dir = wind_par

        # Stochastic parameters
        self.wind_noise_std = 0.2 # wind magnitude noise
        self.wind_dir_noise_std = 5.0 # wind direction noise
        self.action_noise_std = 0.1 # action noise
        self.spray_noise_std = 0.05 # spray noise
        self.obs_noise_std = 0.01 # observation noise
        self.init_position_noise = 0.5 # initial position noise

        # Action space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_robots, 3), dtype=np.float32
        )

        # Observation space
        obs_dim = num_robots * 5 + len(self.infected_levels_init) # = 2*num_robots + 2*num_robots + num_robots + M = 5*num_robots + M
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset() # initialize state

    def _get_obs(self):
        info = {
            f'robot{i}': {
                'position': self.robot_positions[i],
                'capacity': self.robot_capacities[i]
            }
            for i in range(self.num_robots)
        }

        infection_norm = self.infected_levels / self.max_infection_level

        state = np.concatenate([
            self.robot_positions.flatten(),
            self.robot_velocities.flatten(),
            self.robot_capacities,
            infection_norm
        ])

        # Observation noise
        noise = np.random.normal(0, self.obs_noise_std, size=state.shape)
        state = state + noise

        return state.astype(np.float32), info

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        # Domain randomization for wind
        self.wind_mag = self.base_wind_mag + np.random.normal(0, self.wind_noise_std)
        self.wind_dir = self.base_wind_dir + np.random.normal(0, self.wind_dir_noise_std)

        # Randomize robot start positions
        self.robot_positions = self.init_robot_positions + np.random.normal(
            0, self.init_position_noise, self.init_robot_positions.shape
        )

        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_capacities = self.init_robot_capacities.copy()

        # Infected locations
        self.infected_positions = self.infected_positions_init.copy()
        self.infected_levels = self.infected_levels_init.copy()

        self.step_count = 0
        self.total_path_length = 0.0
        self.prev_positions = self.robot_positions.copy()
        self.trajectories = [[] for _ in range(self.num_robots)] # Only for rendering

        return self._get_obs()

    def step(self, actions):
        self.step_count += 1
        reward = 0
        terminated, truncated = False, False

        # Stochastic wind
        wind_mag = self.wind_mag + np.random.normal(0, self.wind_noise_std)
        wind_dir = self.wind_dir + np.random.normal(0, self.wind_dir_noise_std)
        theta = np.radians(wind_dir)
        wind = np.array([wind_mag * np.cos(theta), wind_mag * np.sin(theta)])

        total_sprayed = 0

        for i in range(self.num_robots):
            ax, ay, spray = actions[i]

            # Action noise
            ax = ax * self.thrust_power + np.random.normal(0, self.action_noise_std)
            ay = ay * self.thrust_power + np.random.normal(0, self.action_noise_std)

            # Velocity update
            self.robot_velocities[i] += np.array([ax, ay]) / self.mass + wind
            self.robot_velocities[i] = np.clip(self.robot_velocities[i], self.min_speed, self.max_speed)

            # Position update
            new_pos = self.robot_positions[i] + self.robot_velocities[i]
            outside = not is_inside_polygon(new_pos, self.poly_vertices)
            if outside:
                reward -= 50
                self.robot_velocities[i] = 0
            else:
                self.robot_positions[i] = new_pos

            # keep only last N points (efficiency) in trajectory
            self.trajectories[i].append(self.robot_positions[i].copy())
            if len(self.trajectories[i]) > 200:
                self.trajectories[i].pop(0)

            # Spray (vectorized + capacity constraint)
            if spray > 0 and self.robot_capacities[i] > 0:
                dists = np.linalg.norm(self.robot_positions[i] - self.infected_positions, axis=1)
                mask = (dists <= self.infected_radius) & (self.infected_levels > 0)

                if np.any(mask):
                    w = np.exp(-(dists[mask] ** 2) / (2 * self.spray_sigma ** 2))

                    spray_effect = spray * w
                    spray_effect += np.random.normal(0, self.spray_noise_std, size=w.shape)
                    spray_effect = np.clip(spray_effect, 0, None)

                    applied = np.minimum(spray_effect, self.infected_levels[mask])
                    total = np.sum(applied)

                    # Capacity constraint
                    if total > self.robot_capacities[i]:
                        applied *= self.robot_capacities[i] / (total + 1e-8)
                        total = self.robot_capacities[i]
                    self.robot_capacities[i] -= total
                    self.infected_levels[mask] -= applied

                    total_sprayed += total

            # Energy penalty
            reward -= 0.1 * (ax**2 + ay**2)

            # Movement penalty
            reward -= 0.3 * np.linalg.norm(self.robot_velocities[i]) # movement penalty

        # Path length update (TRUE metric)
        step_dist = np.linalg.norm(self.robot_positions - self.prev_positions, axis=1)
        step_path = np.sum(step_dist)
        self.total_path_length += step_path
        self.prev_positions = self.robot_positions.copy()

        # Spray reward
        reward += 100.0 * total_sprayed
        reward -= 1.0 * step_path   # TRUE path optimality

        # Time penalty
        reward -= 2.0        

        # Remaining infection penalty
        remaining = np.sum(self.infected_levels)
        reward -= 0.5 * remaining               
        if remaining <= 0.01:
            reward += 500.0
            terminated = True
        truncated = self.step_count >= self.max_steps

        # Collision penalty
        if self.num_robots > 1:
            if compute_min_dist(self.robot_positions) < self.robot_size:
                reward -= 1000.0
                terminated = True

        obs, info = self._get_obs()
        info.update({
            "total_sprayed": float(total_sprayed),
            "remaining_infection": float(remaining),
            "episode_length": self.step_count,
            "path_length": float(self.total_path_length)
        })

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.WIDTH * self.render_scale, self.HEIGHT * self.render_scale)
            )
            pygame.display.set_caption("Multi-Robot Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # --- Draw field ---
        scaled_poly = [(x * self.render_scale, y * self.render_scale) for x, y in self.poly_vertices]
        pygame.draw.polygon(self.screen, (255, 255, 0), scaled_poly)

        # --- Draw trajectories ---
        for i in range(self.num_robots):
            if len(self.trajectories[i]) > 1:
                points = [
                    (int(p[0] * self.render_scale), int(p[1] * self.render_scale))
                    for p in self.trajectories[i]
                ]
                pygame.draw.lines(self.screen, (150, 150, 150), False, points, 2)

        # --- Draw robots ---
        colors = [
            (255, 0, 0), (0, 255, 0), (255, 0, 255),
            (255, 128, 0), (128, 0, 255), (255, 0, 255)
        ]

        for i in range(self.num_robots):
            pos = (
                int(self.robot_positions[i][0] * self.render_scale),
                int(self.robot_positions[i][1] * self.render_scale)
            )
            pygame.draw.circle(self.screen, colors[i % len(colors)], pos, 6)

        # --- Draw infected locations (dynamic color) ---
        for pos, level in zip(self.infected_positions, self.infected_levels):
            intensity = int(255 * min(level / 5.0, 1.0))
            scaled_loc = (
                int(pos[0] * self.render_scale),
                int(pos[1] * self.render_scale)
            )
            pygame.draw.circle(self.screen, (0, intensity, 255), scaled_loc, 6)

        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.screen:
            pygame.quit()

# ================================
# TRAINING
# ================================
device_id = "cuda:1"
time_steps = int(1e6)
num_envs = 4
num_robots = 3
max_steps = 1000
method = 5

gym.register(id='MultiRobotEnv-v0', entry_point=MultiRobotEnv, max_episode_steps=max_steps)

json_path = os.path.join('.', 'exp_sets', 'new_2026_cont_sets.json')
json_dict = load_experiment_dict_json(json_path)

log_path = os.path.join(os.getcwd(), f"logs_march26_method{method}", datetime.datetime.now().strftime("%B%d_%H"))
os.makedirs(log_path, exist_ok=True)

for i in range(1, 11):
    vec_env = make_vec_env(
        'MultiRobotEnv-v0',
        env_kwargs={'field_info': json_dict[f"set{i}"], 
                    'num_robots': num_robots, 
                    'max_steps': max_steps, 
                    'render_mode': None},
        n_envs=num_envs
    )

    eval_env = make_vec_env(
        'MultiRobotEnv-v0',
        env_kwargs={
            'field_info': json_dict[f"set{i}"],
            'num_robots': num_robots,
            'max_steps': max_steps,
            'render_mode': None},
        n_envs=1  # usually 1 for evaluation
    )
    
    callback1 = LogEveryNTimesteps(n_steps=10000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, f"best_model_method{method}_env{i}"),
        log_path=os.path.join(log_path, f"eval_logs_env{i}"),
        eval_freq= max(10000 // num_envs, 1),              # evaluate every N steps
        n_eval_episodes=5,            # average over episodes
        deterministic=True,
        render=False
    )
    callback = CallbackList([callback1, eval_callback])

    logger = configure(os.path.join(log_path, f"crossq_env{i}"),
                       ["stdout", "log", "csv", "tensorboard"])

    model = CrossQ("MlpPolicy", vec_env, verbose=1, device=device_id)

    print(f"Training env {i}")
    model.set_logger(logger)
    model.learn(total_timesteps=time_steps, callback=callback)
    model.save(os.path.join(log_path, f"method{method}_env{i}_CrossQ"))

    # model.save(os.path.join(log_path, f"env{i}_CrossQ"))
    del model, vec_env, eval_env