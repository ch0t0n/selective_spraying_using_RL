import pygame
import gymnasium as gym
import numpy as np
from src.utils import is_inside_polygon, compute_min_dist

# ================================================================
# MultiRobotEnv — unified environment supporting:
#
#   reward_ablation  : "full" | "no_term" | "no_spr" | "no_path"
#       "full"       → all reward terms active (default)
#       "no_term"    → collision penalty and success bonus disabled (still terminates episodes)
#       "no_spr"     → spraying terms disabled:
#                      remaining-infection penalty, useless spray penalty disabled 
#                      (but spray bonus remains since it is the most important part)
#       "no_path"    → path terms disabled:
#                      energy penalty, per-robot speed penalty,
#                      path penalty, time penalty
#
#   obs_mode         : "full" | "no_pos" | "no_inf_hist" | "pos_only"
#       "full"       → original obs: positions + velocities +
#                      capacities + infection levels  (5N+M)
#       "no_pos"     → capacities + infection levels (N+M)
#       "no_inf_hist" → positions + velocities + capacities  (5N)
#       "pos_only"   → robot positions only  (2N)
#
#   uncertainty_mode : "full" | "wind_only" | "act_only" | "deterministic"
#       "full"       → all noise sources active (default)
#       "wind_only"  → only wind noise; action + spray noise = 0
#       "act_only"   → only actuation noise; wind + spray noise = 0
#       "deterministic" → all noise sources = 0
#       Note: "wind_only", "act_only", & "deterministic" still have randomness at the episode level in reset() - Randomise starting positions
#
#   dr_mode          : "none" | "wind" | "full"
#       "none"       → no domain randomization (default)
#       "wind"       → re-sample wind speed & direction each episode
#       "full"       → re-sample wind + actuation noise std +
#                      spray radius + UAV mass + thrust scaling
#       Note: "none" & "wind" still have randomness at the episode level in reset() - wind_mag & wind_dir
#
# All parameters default to the original behaviour so existing code
# that does not pass these arguments continues to work unchanged.
# ================================================================

class MultiRobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 60}
    def __init__(
        self,
        field_info,
        render_mode=None,
        wind_par=[0, 0],
        num_robots=3,
        render_scale=5,
        max_steps=1000,
        # ── experiment control ──────────────────────────────────────
        reward_ablation="full",
        obs_mode="full",
        uncertainty_mode="full",
        dr_mode="none",
    ):
        super().__init__()

        # Validate experiment parameters
        assert reward_ablation in ("full", "no_term" , "no_spr" , "no_path"), \
            f"Unknown reward_ablation: {reward_ablation}"
        assert obs_mode in ("full" , "no_pos" , "no_inf_hist" , "pos_only"), \
            f"Unknown obs_mode: {obs_mode}"
        assert uncertainty_mode in ("full" , "wind_only" , "act_only" , "deterministic"), \
            f"Unknown uncertainty_mode: {uncertainty_mode}"
        assert dr_mode in ("none", "wind", "full"), \
            f"Unknown dr_mode: {dr_mode}"
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Store experiment flags
        self.reward_ablation  = reward_ablation
        self.obs_mode         = obs_mode
        self.uncertainty_mode = uncertainty_mode
        self.dr_mode          = dr_mode

        # ── Field info ──────────────────────────────────────────────
        self.field_info = field_info
        self.poly_vertices = field_info['field']
        self.xs, self.ys = zip(*self.poly_vertices)
        self.min_x, self.max_x = np.min(self.xs), np.max(self.xs)
        self.min_y, self.max_y = np.min(self.ys), np.max(self.ys)
        self.world_width  = self.max_x - self.min_x
        self.world_height = self.max_y - self.min_y

        # Simple velocity damping (<1.0 adds friction; can improve stability/learnability)
        self.velocity_damping = 1.0 # 0.96
        # Multiplier to robot velocity on boundary violation (0.0 = stop)
        self.boundary_violation_damping = 0.0 # 0.25

        # ── Rendering ───────────────────────────────────────────────
        self.render_scale = render_scale
        self.screen_width, self.screen_height = 800, 800
        self.offset_x = (self.screen_width  - self.world_width  * self.render_scale) / 2
        self.offset_y = (self.screen_height - self.world_height * self.render_scale) / 2
        self.max_steps = max_steps        
        self.render_mode = render_mode
        self.screen = None
        self.clock  = None

        # ── Robot params ────────────────────────────────────────────
        self.num_robots = num_robots
        self.init_robot_positions = np.array(
            field_info['init_positions'][:num_robots], dtype=np.float32)
        self.init_robot_capacities = np.array(
            field_info['robot_capacities'][:num_robots], dtype=np.float32)
        self.robot_size   = 2    # collision radius
        self.mass         = 1.0  # UAV mass (overridden by DR full)
        self.thrust_power = 0.5  # action scaling (overridden by DR full)
        self.max_speed    = 5
        self.min_speed    = -5
        self.robot_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Lime / Green
            (255, 0, 255),  # Magenta / Fuchsia
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Violet / Purple
            (255, 0, 255)   # Magenta / Fuchsia
        ]

        # ── Infection params ────────────────────────────────────────
        self._nominal_infected_radius = 5   # nominal spray radius (DR may override per-episode)
        self.infected_radius = self._nominal_infected_radius
        self.spray_sigma     = self.infected_radius / 2
        self.init_infected_positions = np.array(
            [[x, y] for x, y, _ in field_info['infected_locations']], dtype=np.float32)
        self.init_infected_levels = np.array(
            [lvl for _, _, lvl in field_info['infected_locations']], dtype=np.float32)
        self.max_infection_level = np.max(self.init_infected_levels)
        self.initial_infection_sum = np.sum(self.init_infected_levels)
        # ── Base wind (mean) magnitude and direction
        self.base_wind_mag, self.base_wind_dir = wind_par

        # ── Noise stds — set by uncertainty_mode ────────────────────
        # These are the *nominal* values; DR "full" may override
        # action_noise_std at the start of each episode.
        _noise = {
            "full":          dict(wind=0.20, wind_dir=5.0, action=0.10, spray=0.05, obs=0.01),
            "wind_only":     dict(wind=0.20, wind_dir=5.0, action=0.00, spray=0.00, obs=0.00),
            "act_only":      dict(wind=0.00, wind_dir=0.0, action=0.10, spray=0.00, obs=0.00),
            "deterministic": dict(wind=0.00, wind_dir=0.0, action=0.00, spray=0.00, obs=0.00),
        }[uncertainty_mode]

        self.wind_noise_std         = _noise["wind"]
        self.wind_dir_noise_std     = _noise["wind_dir"]
        self.action_noise_std       = _noise["action"]
        self.spray_noise_std        = _noise["spray"]
        self.obs_noise_std          = _noise["obs"]
        self.init_position_noise    = 0.5

        # ── Nominal noise stds (DR "full" samples around these) ─────
        self._nominal_action_noise_std = _noise["action"]
        self._nominal_mass             = 1.0
        self._nominal_thrust_power     = 0.5

        # Each robot's action have the following:
        #   1. a_x = Force component (or thrust) along x-axis
        #   2. a_y = Force component (or thrust) along y-axis
        #   3. sigma = spray rate
        # Therefore, total actions = 3*num_robots

        # ── Action space: (ax, ay, spray) per robot ─────────────────
        self.action_space = gym.spaces.Box(
            low=np.array([[-1.0, -1.0, 0.0]] * self.num_robots, dtype=np.float32),
            high=np.array([[1.0, 1.0, 1.0]] * self.num_robots, dtype=np.float32),
            dtype=np.float32,
        )

        # Each robot's observation have the following for base:
        #    1. (x, y) co-ordinates of each robot
        #    2. (v_x, v_y) velocities along x and y axis of each robot
        #    3. current spraying capacity (C_p(t))
        #    4. current infection levels of M locations
        # Therefore, total observations = 2*num_robots (positions) + 2*num_robots (velocities) + 
        # num_robots (spraying capacities) + M (num_inf_locations) = 5*num_robots + M
        # Used to normalize coordinates into the field bounding box.
        self._coord_offset = np.array([self.min_x, self.min_y], dtype=np.float32)
        self._coord_norm = np.maximum(
            np.array([self.world_width, self.world_height], dtype=np.float32),
            np.finfo(np.float32).eps,
        )
        
        # ── Observation space — depends on obs_mode ──────────────────
        M = len(self.init_infected_levels)
        N = num_robots
        _obs_dims = {
            "full":          5*N + M,       # original
            "no_pos":        N + M,         # capacities(N) + inf_levels(M)
            "no_inf_hist":   5*N,     # 5N 
            "pos_only":      2*N,           # positions only
        }
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(_obs_dims[obs_mode],), dtype=np.float32)

        self.reset()  # initialise state

    # ── coordinate conversion ────────────────────────────────────────
    def world_to_screen(self, pos):
        x = (pos[0] - self.min_x) * self.render_scale + self.offset_x
        y = (pos[1] - self.min_y) * self.render_scale + self.offset_y
        return int(x), int(y)

    # ── observation builder ──────────────────────────────────────────
    def _get_obs(self):
        info = {
            f'robot{i}': {
                'position': self.robot_positions[i],
                'capacity': self.robot_capacities[i],
            }
            for i in range(self.num_robots)
        }
        infection_norm = self.infected_levels / self.max_infection_level
        robot_positions_normalized = (self.robot_positions - self._coord_offset) / self._coord_norm

        if self.obs_mode == "full":
            state = np.concatenate([robot_positions_normalized.flatten(),
            self.robot_velocities.flatten(),
            self.robot_capacities,
            infection_norm])
        elif self.obs_mode == "no_pos":
            state = np.concatenate([self.robot_capacities, infection_norm])
        elif self.obs_mode == "no_inf_hist":
            state = np.concatenate([robot_positions_normalized.flatten(),
            self.robot_velocities.flatten(),
            self.robot_capacities])
        elif self.obs_mode == "pos_only":
            state = robot_positions_normalized.flatten().copy()

        if self.obs_noise_std > 0:
            state = state + self.np_random.normal(0, self.obs_noise_std, size=state.shape)

        return state.astype(np.float32), info

    # ── reset ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        super().reset(seed=seed)

        # ── Domain randomization — re-sample base params each episode ─
        if self.dr_mode == "none":
            # Standard: add small episode-level noise to the nominal wind
            self.wind_mag = self.base_wind_mag + self.np_random.normal(0, self.wind_noise_std)
            self.wind_dir = self.base_wind_dir + self.np_random.normal(0, self.wind_dir_noise_std)
            # Restore nominal physical params (may have been overridden last episode)
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_radius  = self._nominal_infected_radius
            self.spray_sigma      = self.infected_radius / 2
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power

        elif self.dr_mode == "wind":
            # Randomise wind speed and direction uniformly
            self.wind_mag = float(self.np_random.uniform(0.0, 1.0))
            self.wind_dir = float(np.degrees(self.np_random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_radius  = self._nominal_infected_radius
            self.spray_sigma      = self.infected_radius / 2
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power

        elif self.dr_mode == "full":
            # Randomise all physical parameters
            self.wind_mag         = float(self.np_random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(self.np_random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = float(self.np_random.uniform(0.01, 0.10))
            r0 = self._nominal_infected_radius
            self.infected_radius  = float(self.np_random.uniform(0.8 * r0, 1.2 * r0))
            self.spray_sigma      = self.infected_radius / 2
            self.mass             = float(self.np_random.uniform(0.9, 1.1))
            self.thrust_power     = 0.5 * float(self.np_random.uniform(0.8, 1.2))

        # ── Randomise starting positions ─────────────────────────────
        self.robot_positions = self.init_robot_positions + self.np_random.normal(
            0, self.init_position_noise, self.init_robot_positions.shape)

        # ── Initialise dynamic state ─────────────────────────────────
        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_capacities = self.init_robot_capacities.copy()
        self.infected_positions = self.init_infected_positions.copy()
        self.infected_positions_normalized = (self.infected_positions - self._coord_offset) / self._coord_norm
        self.infected_levels    = self.init_infected_levels.copy()

        # ── Episode counters ─────────────────────────────────────────
        self.step_count       = 0
        self.total_path_length = 0.0
        self.prev_positions   = self.robot_positions.copy()
        self.trajectories     = [[] for _ in range(self.num_robots)] # Only for rendering

        self._reset_episode_metrics()
        return self._get_obs()

    # ── step ─────────────────────────────────────────────────────────
    def step(self, actions):
        self.step_count += 1
        reward = 0
        terminated, truncated = False, False

        # ── Stochastic wind for this step ────────────────────────────
        wind_mag = self.wind_mag + self.np_random.normal(0, self.wind_noise_std)
        wind_dir = self.wind_dir + self.np_random.normal(0, self.wind_dir_noise_std)
        theta = np.radians(wind_dir)
        wind  = np.array([wind_mag * np.cos(theta), wind_mag * np.sin(theta)])

        total_sprayed = 0                       # Track total infection reduced this step (for reward)
        for i in range(self.num_robots):        # For each robot
            ax, ay, spray = actions[i]          # Get actual actions: thrust_x, thrust_y, sigma (spray amount)

            # Action noise + scaling
            ax = ax * self.thrust_power + self.np_random.normal(0, self.action_noise_std)
            ay = ay * self.thrust_power + self.np_random.normal(0, self.action_noise_std)

            # Velocity update (Newtonian dynamics)
            self.robot_velocities[i] += np.array([ax, ay]) / self.mass + wind
            self.robot_velocities[i]  = np.clip(
                self.robot_velocities[i], self.min_speed, self.max_speed)

            # Position update
            new_pos = self.robot_positions[i] + self.robot_velocities[i] # Predict next position
            outside = not is_inside_polygon(new_pos, self.poly_vertices) # Check if new position is outside the field
            if outside:                                 # If new position is outside the field
                reward -= 50                            # Penalty for moving outside the field
                self.robot_velocities[i] *= self.boundary_violation_damping
                self.episode_boundary_viol_count += 1
            else:
                self.robot_positions[i] = new_pos       # Else, move to the new location

            # Metrics (per step, per robot)
            step_spray_used_i = 0.0
            step_spray_applied_i = 0.0   # 'successful' spray
            step_spray_wasted_i = 0.0    # 'wasted' spray i.e. spray missed infection - sprayed healthy areas and wasted resources
            empty_capacity_spray_i = 0.0 # spray action > capacity (useful?)
            # Spray dynamics (vectorized + capacity constraint)
            available_capacity = float(self.robot_capacities[i])
            if spray > 0: 
                if available_capacity <= 0.0:
                    empty_capacity_spray_i = spray                               # All spray command is useless since no capacity left
                else:                                                            # Only spray if spray rate > 0 and remaining spraying capacities > 0
                    step_spray_used_i = min(spray, available_capacity)
                    empty_capacity_spray_i = max(0.0, spray - step_spray_used_i) # Amount of spray that is beyond the available capacity (useless since it cannot be applied)   

                    dists = np.linalg.norm(
                        self.robot_positions[i] - self.infected_positions, axis=1)          # Distances from robot to infected locations
                    mask  = (dists <= self.infected_radius) & (self.infected_levels > 0)    # Only affect nearby infected locations

                    if np.any(mask):                                                                # If nearby infected locations exists
                        w = np.exp(-(dists[mask] ** 2) / (2 * self.spray_sigma ** 2))               # Gaussian decay: closer points get more spray effect
                        spray_effect = spray * w                                                    # Scale spray intensity by distance weighting
                        spray_effect += self.np_random.normal(0, self.spray_noise_std, size=w.shape)     # Add stochasticity to spraying process
                        spray_effect  = np.clip(spray_effect, 0, None)                              # Prevent negative spray
                        applied       = np.minimum(spray_effect, self.infected_levels[mask])        # Cannot disinfect more infection than exists
                        total         = np.sum(applied)                                             # Total spray to apply
                        if total > self.robot_capacities[i]:                        # If exceeding available spray capacity
                            applied *= self.robot_capacities[i] / (total + 1e-8)    # Scale down proportionally
                            total    = self.robot_capacities[i]                     # Total spray applied by this robot in this step
                        self.robot_capacities[i]    -= total                        # Reduce sprayed amount from capacity
                        self.infected_levels[mask]  -= applied                      # Reduce infection levels by applied spray amount
                        total_sprayed               += total                        # Accumulate global spraying amount for all robots
                        step_spray_applied_i         = float(total)
                    else:
                        # Useless spray penalty (part of R_eff)
                        if self.reward_ablation != "no_spr":
                            reward -= 0.05 * spray
                    step_spray_wasted_i = max(0.0, step_spray_used_i - step_spray_applied_i)

            # Apply velocity damping (friction) after movement
            self.robot_velocities[i] *= self.velocity_damping

            # Update spray history for this robot
            self.episode_spray_used += step_spray_used_i
            self.episode_spray_applied += step_spray_applied_i
            self.episode_spray_wasted += step_spray_wasted_i
            self.episode_spray_empty_capacity += empty_capacity_spray_i

            # ── Per-robot movement penalties ─────────────────
            if self.reward_ablation != "no_path":
                reward -= 0.1 * (ax ** 2 + ay ** 2)                       # Energy penalty (encourages efficient control)
                reward -= 0.3 * np.linalg.norm(self.robot_velocities[i])  # Movement penalty (Penalize high speeds)

            # Trajectory buffer (rendering only)
            self.trajectories[i].append(self.robot_positions[i].copy())
            if len(self.trajectories[i]) > 200:
                self.trajectories[i].pop(0)

        # Path length computation (true traveled distance)
        step_dist = np.linalg.norm(self.robot_positions - self.prev_positions, axis=1)   # Distance traveled by each robot this step
        step_path = np.sum(step_dist)                                                    # Total distance traveled by all robots
        self.total_path_length += step_path                                              # Accumulate episode path length
        self.prev_positions = self.robot_positions.copy()                                # Update previous positions

        # ── R_spray: core spraying reward ────────────────────────────
        if self.reward_ablation != "no_spr":
            reward += 100.0 * total_sprayed     # Strong positive reward for removing infection

        # ── R_eff: global efficiency penalties ───────────────────────
        if self.reward_ablation != "no_path":
            reward -= 1.0 * step_path       # path length penalty
            reward -= 2.0                   # time penalty

        # Remaining infection penalty
        remaining = np.sum(self.infected_levels)
        if self.reward_ablation != "no_spr":
            # Distance shaping to nearest active infection
            active_mask = self.infected_levels > 0.01
            if np.any(active_mask):
                active_pos = self.infected_positions[active_mask]
                dists_mat  = np.linalg.norm(
                    self.robot_positions[:, None] - active_pos[None, :], axis=2)
                nearest    = np.min(dists_mat, axis=1)
                reward    += 0.5 * np.sum(np.exp(-nearest))            
            reward -= 0.5 * remaining

        term_cond = ""               # Terminal conditions for the reward ablation
        if remaining <= 0.01:
            if self.reward_ablation != "no_term":
                reward += 500.0      # Large positive reward for successfull spraying
            term_cond = "sprayed"   
            terminated = True        # always terminate on succession (completion)

        # ── R_col: collision penalty ──────────────────────────────────
        if self.num_robots > 1:                                             # Check collision only for multi-robots
            if compute_min_dist(self.robot_positions) < self.robot_size:    # Robots too close
                if self.reward_ablation != "no_term":
                    reward -= 1000.0                                        # Large negative reward for collision
                term_cond = "collision"
                terminated = True   # always terminate on collision (safety)

        # ── Build observation & info ──────────────────────────────────
        obs, info = self._get_obs()
        truncated = self.step_count >= self.max_steps       # Episode ends if max steps reached
        if truncated:
            term_cond = "max_steps"
        info.update({
            "total_sprayed": float(total_sprayed),          # Total infection removed this step
            "remaining_infection": float(remaining),        # Remaining infected amount
            "episode_length": self.step_count,              # Current timestep
            "path_length": float(self.total_path_length),   # Total distance traveled
            "term_cond": term_cond
        })

        if terminated or truncated:
            info['episode_metrics'] = self._get_episode_metrics(term_cond)
        return obs, reward, terminated, truncated, info

    # ── render ───────────────────────────────────────────────────────
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

    
    def _reset_episode_metrics(self):
        self.episode_spray_used = 0.0 # Consumed spray
        self.episode_spray_applied = 0.0 # Successful spray
        self.episode_spray_wasted = 0.0 # Wasted spray
        self.episode_spray_empty_capacity = 0.0 # Spray commanded beyond available capacity
        self.episode_wasted_fraction = 0.0
        self.episode_boundary_viol_count = 0
        self.episode_remaining_infection_sum = 0.0
        self.episode_remaining_infection_fraction = 0.0
        self.episode_time_limit_reached = False
        self.episode_collision_occurred = False
        self.episode_success = False

    def _get_episode_metrics(self, term_cond):
        if term_cond == "collision":
            self.episode_collision_occurred = True
        elif term_cond == "sprayed":
            self.episode_success = True
        elif term_cond == "max_steps":
            self.episode_time_limit_reached = True
        else:
            pass # This should not happen
        if self.episode_spray_used > 0:
            self.episode_wasted_fraction = float(
                np.clip(self.episode_spray_wasted / self.episode_spray_used, 0.0, 1.0)
            )
        else:
            self.episode_wasted_fraction = 0.0
        self.episode_remaining_infection_sum = float(np.sum(np.clip(self.infected_levels, 0.0, None)))
        initial_sum = max(float(self.initial_infection_sum), 1e-8)
        self.episode_remaining_infection_fraction = float(
            np.clip(self.episode_remaining_infection_sum / initial_sum, 0.0, 1.0)
        )
        return {
            "episode_step_count": int(self.step_count),
            "episode_spray_used": float(self.episode_spray_used),
            "episode_spray_wasted": float(self.episode_spray_wasted),
            "episode_spray_applied": float(self.episode_spray_applied),
            "episode_wasted_fraction": float(self.episode_wasted_fraction),
            "episode_spray_empty_capacity": float(self.episode_spray_empty_capacity),
            "episode_boundary_viol_count": int(self.episode_boundary_viol_count),
            "episode_remaining_infection_sum": float(self.episode_remaining_infection_sum),
            "episode_remaining_infection_fraction": float(self.episode_remaining_infection_fraction),
            "episode_time_limit_reached": bool(getattr(self, 'episode_time_limit_reached', False)),
            "episode_collision_occurred": bool(self.episode_collision_occurred),
            "episode_success": bool(self.episode_success)
        }
