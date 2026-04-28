# ================================
# This is the version 1.0 of the final spraying env confirmed in April 8, 2026
# ================================

import os  # OS operations (paths, folders)
import json  # JSON file handling
from datetime import datetime  # timestamping logs
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


def load_experiment_dict_json(json_path):  # load experiment configs
    with open(json_path, "r") as f:  # open file
        data = json.load(f)  # load JSON
    for _, cfg in data.items():  # iterate over configs
        cfg["field"] = [tuple(p) for p in cfg["field"]]  # convert to tuples
        cfg["init_positions"] = np.array(cfg["init_positions"], dtype=float)  # positions array
        cfg["infected_locations"] = [tuple(p) for p in cfg["infected_locations"]]  # infection
    return data  # return processed dict


# ================================
# Environment
# ================================

class MultiRobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 60}
    def __init__(self, field_info, render_mode=None, wind_par=[0, 0], num_robots=3, render_scale=5, max_steps=1000):
        super().__init__()

        # Field info
        self.field_info = field_info
        self.poly_vertices = field_info['field']
        self.xs, self.ys = zip(*self.poly_vertices) # separate coordinates
        self.min_x, self.max_x = np.min(self.xs), np.max(self.xs)
        self.min_y, self.max_y = np.min(self.ys), np.max(self.ys)
        self.world_width = self.max_x - self.min_x
        self.world_height = self.max_y - self.min_y
        
        # Rendering info
        self.render_scale = render_scale # scaling for rendering
        self.screen_width, self.screen_height = 800, 800
        self.offset_x = (self.screen_width - self.world_width * self.render_scale) / 2
        self.offset_y = (self.screen_height - self.world_height * self.render_scale) / 2
        self.max_steps = max_steps # max episode length
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Robot params
        self.num_robots = num_robots
        self.init_robot_positions = field_info['init_positions'][:num_robots]
        self.init_robot_capacities = np.array(field_info['robot_capacities'][:num_robots], dtype=np.float32)
        self.robot_size = 2 # collision radius
        self.mass = 1.0 # robot mass
        self.thrust_power = 0.5 # action scaling
        self.max_speed = 5
        self.min_speed = -5        
        self.robot_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Lime / Green
            (255, 0, 255),  # Magenta / Fuchsia
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Violet / Purple
            (255, 0, 255)   # Magenta / Fuchsia
        ]

        # Infection params
        self.infected_radius = 5 # spray radius
        self.spray_sigma = self.infected_radius / 2 # Gaussian spread
        self.infected_positions_init = np.array([[x, y] for x, y, _ in field_info['infected_locations']])
        self.infected_levels_init = np.array([lvl for _, _, lvl in field_info['infected_locations']], dtype=np.float32)
        self.max_infection_level = np.max(self.infected_levels_init)

        # Base wind (mean) magnitude and direction
        self.base_wind_mag, self.base_wind_dir = wind_par

        # Stochastic parameters
        self.wind_noise_std = 0.2 # wind magnitude noise
        self.wind_dir_noise_std = 5.0 # wind direction noise
        self.action_noise_std = 0.1 # action noise
        self.spray_noise_std = 0.05 # spray noise
        self.obs_noise_std = 0.01 # observation noise
        self.init_position_noise = 0.5 # initial position noise

        # Each robot's action have the following:
        #   1. a_x = Force component (or thrust) along x-axis
        #   2. a_y = Force component (or thrust) along y-axis
        #   3. sigma = spray rate
        # Therefore, total actions = 3*num_robots

        # Action space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_robots, 3), dtype=np.float32
        )

        # Each robot's observation have the following:
        #    1. (x, y) co-ordinates of each robot
        #    2. (v_x, v_y) velocities along x and y axis of each robot
        #    3. current spraying capacity (C_p(t))
        #    4. current infection levels of M locations
        # Therefore, total observations = 2*num_robots (positions) + 2*num_robots (velocities) + 
        # num_robots (spraying capacities) + M (num_inf_locations) = 5*num_robots + M

        # Observation space
        obs_dim = num_robots * 5 + len(self.infected_levels_init)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )        

        self.reset() # initialize state

    def world_to_screen(self, pos):  # convert world coordinates to screen coordinates
            x = (pos[0] - self.min_x) * self.render_scale + self.offset_x
            y = (pos[1] - self.min_y) * self.render_scale + self.offset_y
            return int(x), int(y)

    def _get_obs(self): # get current observation
        info = {
            f'robot{i}': {
                'position': self.robot_positions[i],
                'capacity': self.robot_capacities[i]
            }
            for i in range(self.num_robots)
        }

        infection_norm = self.infected_levels / self.max_infection_level # normalize infection levels

        state = np.concatenate([            # current observation (or state)
            self.robot_positions.flatten(),
            self.robot_velocities.flatten(),
            self.robot_capacities,
            infection_norm
        ])

        # Observation noise
        noise = np.random.normal(0, self.obs_noise_std, size=state.shape)
        state = state + noise # updated state with noise

        return state.astype(np.float32), info

    def reset(self, seed=None, options={}): # reset and initialize starting state
        super().reset(seed=seed)

        # Domain randomization for wind
        self.wind_mag = self.base_wind_mag + np.random.normal(0, self.wind_noise_std)
        self.wind_dir = self.base_wind_dir + np.random.normal(0, self.wind_dir_noise_std)

        # Randomize robot start positions
        self.robot_positions = self.init_robot_positions + np.random.normal(
            0, self.init_position_noise, self.init_robot_positions.shape
        )

        # Initialize robot velocities and spraying capacities
        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_capacities = self.init_robot_capacities.copy()

        # Initialize infected locations
        self.infected_positions = self.infected_positions_init.copy()
        self.infected_levels = self.infected_levels_init.copy()

        # Initialize step count, path length and empty trajectories
        self.step_count = 0
        self.total_path_length = 0.0
        self.prev_positions = self.robot_positions.copy()
        self.trajectories = [[] for _ in range(self.num_robots)] # Only for rendering

        return self._get_obs()

    def step(self, actions):
        self.step_count += 1                    # Increase timestep counter (used for truncation and logging)
        reward = 0                              # Initialize cumulative reward for this step
        terminated, truncated = False, False    # Flags: terminated = success/failure, truncated = max steps reached

        # Stochastic wind
        wind_mag = self.wind_mag + np.random.normal(0, self.wind_noise_std)     # Add noise to wind magnitude
        wind_dir = self.wind_dir + np.random.normal(0, self.wind_dir_noise_std) # Add noise to wind direction (degrees)
        theta = np.radians(wind_dir)                                            # Convert wind direction from degrees to radians
        wind = np.array([wind_mag * np.cos(theta), wind_mag * np.sin(theta)])   # Convert polar wind (magnitude, angle) to Cartesian vector

        total_sprayed = 0                       # Track total infection reduced this step (for reward)
        for i in range(self.num_robots):        # For each robot
            ax, ay, spray = actions[i]          # Get actual actions: thrust_x, thrust_y, sigma (spray amount)

            # Action noise + scaling
            ax = ax * self.thrust_power + np.random.normal(0, self.action_noise_std) # Scale + noise on x-thrust
            ay = ay * self.thrust_power + np.random.normal(0, self.action_noise_std) # Scale + noise on y-thrust

            # Velocity update (Newtonian dynamics)
            self.robot_velocities[i] += np.array([ax, ay]) / self.mass + wind
            self.robot_velocities[i] = np.clip(self.robot_velocities[i], self.min_speed, self.max_speed) # Enforce speed limits for stability

            # Position update
            new_pos = self.robot_positions[i] + self.robot_velocities[i] # Predict next position
            outside = not is_inside_polygon(new_pos, self.poly_vertices) # Check if new position is outside the field
            if outside:                             # If new position is outside the field
                reward -= 50                        # Penalty for moving outside the field
                self.robot_velocities[i] = 0        # Set all robot velocities to zero
            else:
                self.robot_positions[i] = new_pos   # Else, move to the new location

            # Spray dynamics (vectorized + capacity constraint)
            if spray > 0 and self.robot_capacities[i] > 0:                                        # Only spray if spray rate > 0 and remaining spraying capacities > 0
                dists = np.linalg.norm(self.robot_positions[i] - self.infected_positions, axis=1) # Distances from robot to infected locations
                mask = (dists <= self.infected_radius) & (self.infected_levels > 0)               # Only affect nearby infected locations

                if np.any(mask):        # If nearby infected locations exists
                    w = np.exp(-(dists[mask] ** 2) / (2 * self.spray_sigma ** 2))           # Gaussian decay: closer points get more spray effect
                    spray_effect = spray * w                                                # Scale spray intensity by distance weighting
                    spray_effect += np.random.normal(0, self.spray_noise_std, size=w.shape) # Add stochasticity to spraying process
                    spray_effect = np.clip(spray_effect, 0, None)                           # Prevent negative spray
                    applied = np.minimum(spray_effect, self.infected_levels[mask])          # Cannot disinfect more infection than exists
                    total = np.sum(applied)                                                 # Total spray to apply

                    # Capacity constraint
                    if total > self.robot_capacities[i]:                                    # If exceeding available spray capacity
                        applied *= self.robot_capacities[i] / (total + 1e-8)                # Scale down proportionally
                        total = self.robot_capacities[i]                                    # Total spray applied by this robot in this step
                    self.robot_capacities[i] -= total                                       # Reduce sprayed amount from capacity
                    self.infected_levels[mask] -= applied                                   # Reduce infection levels by applied spray amount
                    total_sprayed += total                                                  # Accumulate global spraying amount for all robots
                else:
                    reward -= 0.05 * spray                                                   # Useless spraying, no nearby infected location

            # Energy penalty (encourages efficient control)
            reward -= 0.1 * (ax**2 + ay**2)

            # Movement penalty (Penalize high speeds)
            reward -= 0.3 * np.linalg.norm(self.robot_velocities[i])

            # (FOR RENDERING ONLY) keep only last N points in trajectory
            self.trajectories[i].append(self.robot_positions[i].copy())
            if len(self.trajectories[i]) > 200:
                self.trajectories[i].pop(0)

        # Path length computation (true traveled distance)
        step_dist = np.linalg.norm(self.robot_positions - self.prev_positions, axis=1)   # Distance traveled by each robot this step
        step_path = np.sum(step_dist)                                                    # Total distance traveled by all robots
        self.total_path_length += step_path                                              # Accumulate episode path length
        self.prev_positions = self.robot_positions.copy()                                # Update previous positions

        # Spray reward
        reward += 100.0 * total_sprayed          # Strong positive reward for removing infection

        # Movement penalty
        reward -= 1.0 * step_path                # Penalize unnecessary movement to keep the path length shortest

        # Distance shaping (only active infections)
        active_mask = self.infected_levels > 0
        if np.any(active_mask):
            active_positions = self.infected_positions[active_mask]
            dists = np.linalg.norm(self.robot_positions[:, None] - active_positions[None, :], axis=2)
            nearest = np.min(dists, axis=1)
            reward += 0.5 * np.sum(np.exp(-nearest))

        # Time penalty
        reward -= 2.0                            # Encourages faster task completion

        # Remaining infection penalty
        remaining = np.sum(self.infected_levels) # Total infection left
        reward -= 0.5 * remaining                # Penalize incomplete cleaning
        if remaining <= 0.01:                    # Success condition (all infection cleared)
            reward += 500.0                     # Large positive reward for successfull spraying
            terminated = True

        # Collision penalty
        if self.num_robots > 1:                                           # Check collision only for multi-robots
            if compute_min_dist(self.robot_positions) < self.robot_size:  # Robots too close
                reward -= 1000.0                                          # Large negative reward for collision
                terminated = True                                         

        obs, info = self._get_obs()                         # Get current observation and basic info
        info.update({
            "total_sprayed": float(total_sprayed),          # Total infection removed this step
            "remaining_infection": float(remaining),        # Remaining infected amount
            "episode_length": self.step_count,              # Current timestep
            "path_length": float(self.total_path_length)    # Total distance traveled
        })
        truncated = self.step_count >= self.max_steps # Episode ends if max steps reached

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human": 
            return              # Only render if mode is set to "human" (skip rendering for training efficiency)

        if self.screen is None:
            pygame.init()                                                                   # Initialize all pygame modules
            pygame.display.init()                                                           # Initialize display module explicitly
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))  # Create window with predefined screen size
            pygame.display.set_caption("Multi-Robot Spraying Environment")                  # Window title
            self.clock = pygame.time.Clock()                                                # Clock to control rendering FPS

        self.screen.fill((255, 255, 255))       # Fill screen with white color

        # --- Draw field boundary ---
        scaled_poly = [self.world_to_screen(p) for p in self.poly_vertices] # Convert field vertices from world coordinates → screen coordinates
        pygame.draw.polygon(self.screen, (255, 255, 0), scaled_poly)

        # --- Draw robot trajectories (paths) ---
        for i in range(self.num_robots):
            if len(self.trajectories[i]) > 1:                                       # Only draw if at least two points exist (needed for a line)
                points = [self.world_to_screen(p) for p in self.trajectories[i]]    # Convert trajectory points to screen coordinates
                pygame.draw.lines(self.screen, (150, 150, 150), False, points, 2)   # Draw polyline showing path history

        # --- Draw robots (current positions) ---        
        for i in range(self.num_robots):
            pos = self.world_to_screen(self.robot_positions[i])                 # Convert robot position to screen coordinates
            pygame.draw.circle(self.screen,
                               self.robot_colors[i % len(self.robot_colors)],   # Assign color cyclically
                               pos, self.robot_size * self.render_scale / 2)    # Scale radius for visualization

        # --- Draw infected locations (dynamic color) ---
        for pos, level in zip(self.infected_positions, self.infected_levels):
            intensity = int(255 * min(level / 5.0, 1.0))                          # Map infection level → color intensity (visual feedback)
            pygame.draw.circle(self.screen, (0, 255-intensity, 255),              # Higher infection → darker blue and lower infection → lighter blue
                               self.world_to_screen(pos), self.render_scale + 2)  # Fixed radius for visibility
        
        # Update display
        pygame.display.flip()  # Push all drawings to the screen (double buffering)
        self.clock.tick(60)    # Limit rendering to 60 FPS for smooth visualization

    def close(self):
        if self.screen:
            pygame.quit()

# ================================
# TRAINING ON CROSSQ ALGORITHM
# ================================

device_id = "cuda:0"
time_steps = int(2e6)
num_envs = 4
num_robots = 3
max_steps = 1000
version = 'v1'

gym.register(id='MultiRobotEnv-v0', entry_point=MultiRobotEnv, max_episode_steps=max_steps)

json_path = os.path.join('..', 'exp_sets', 'stochastic_envs_v2.json')
json_dict = load_experiment_dict_json(json_path)

log_path = os.path.join('..', 'logs', f"{datetime.now().strftime('%b%d')}_version_{version}")
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
        n_envs=1
    )

    callback1 = LogEveryNTimesteps(n_steps=10000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, f"best_model_version_{version}_env{i}"),
        log_path=os.path.join(log_path, f"eval_logs_env{i}"),
        eval_freq=max(10000 // num_envs, 1),
        n_eval_episodes=5,
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
    model.save(os.path.join(log_path, f"version_{version}_env{i}_CrossQ"))

    del model, vec_env, eval_env