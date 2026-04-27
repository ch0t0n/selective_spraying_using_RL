"""
Table 8 — Observation-gap analysis (sim-to-real).

INSTRUCTIONS:
  1. Set RENDER_COPPELIA = False for automated 50-episode batch runs.
  2. Set RENDER_COPPELIA = True only for one-episode visual inspection
     (requires CoppeliaSim scene file open first).
  3. After running, copy the printed IQM values into tab:obs_gap in full_experiments.tex.

Policy: CrossQ + full DR  (obs_mode="full", dr_mode="full")
Env:    variation 1, N=3

obs_mode="full" layout  (5N + M):
    [0    : 2N]     robot positions  (x, y per robot)
    [2N   : 4N]     robot velocities (vx, vy per robot)
    [4N   : 5N]     spray capacities
    [5N   : 5N+M]   infection levels (normalised)

Perturbation → obs slice targeted:
    GPS noise          → positions  [0 : 2N]         add N(0, σ=5.0)
    Wind latency       → velocities [2N : 4N]        replace with 5-step-old values
    Comm. packet loss  → infection  [5N : 5N+M]      zero with prob 0.10
"""

import os, json
from collections import deque
import numpy as np
import gymnasium as gym
import pygame
from sb3_contrib import CrossQ
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
TRAINED_MODEL_PATH = r"/homes/choton/rl4pag/neurips_experiments_v5/logs/main_default/CrossQ_N3_env1_seed42/CrossQ_N3_env1.zip"
JSON_PATH          = os.path.join('.', 'exp_sets', 'stochastic_envs_v2.json')
ENV_VARIATION      = 1
NUM_ROBOTS         = 3
N_EVAL_EPISODES    = 50
RENDER_COPPELIA    = False   # False = pure Python batch eval; True = CoppeliaSim visual
HEIGHT             = 0.4


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────
def load_experiment_dict_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    for _, cfg in data.items():
        cfg["field"]              = [tuple(p) for p in cfg["field"]]
        cfg["init_positions"]     = np.array(cfg["init_positions"], dtype=float)
        cfg["infected_locations"] = [tuple(p) for p in cfg["infected_locations"]]
    return data


def compute_iqm(rewards):
    rewards = np.array(rewards, dtype=np.float32)
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))


# ────────────────────────────────────────────────────────────────
# Observation perturber
#
# obs_mode="full"  →  layout: [pos | vel | cap | inf]
#   [0    : 2N]   positions
#   [2N   : 4N]   velocities
#   [4N   : 5N]   capacities
#   [5N   : 5N+M] infection levels
# ────────────────────────────────────────────────────────────────
class ObsPerturber:
    def __init__(self, N: int, M: int,
                 gps_noise: float        = 0.0,
                 wind_latency_steps: int = 0,
                 packet_loss_prob: float = 0.0):
        self.N = N
        self.M = M
        self.gps_noise          = gps_noise
        self.wind_latency_steps = wind_latency_steps
        self.packet_loss_prob   = packet_loss_prob

        # index boundaries
        self._pos_s = 0
        self._pos_e = 2 * N          # positions
        self._vel_s = 2 * N
        self._vel_e = 4 * N          # velocities
        self._inf_s = 5 * N
        self._inf_e = 5 * N + M      # infection levels

        # N-step velocity deque for wind latency.
        # Oldest entry (index 0) is what the agent sees; newest is current.
        # Initialised to zeros so pre-episode steps are treated as still.
        self._vel_buf: deque = deque(
            [np.zeros(2 * N, dtype=np.float32)] * wind_latency_steps,
            maxlen=wind_latency_steps if wind_latency_steps > 0 else 1,
        )

    def reset(self):
        """Call at the start of every episode."""
        for i in range(len(self._vel_buf)):
            self._vel_buf[i] = np.zeros(2 * self.N, dtype=np.float32)

    def apply(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()

        # ── GPS noise: corrupt position components ───────────────────
        if self.gps_noise > 0.0:
            obs[self._pos_s : self._pos_e] += np.random.normal(
                0.0, self.gps_noise,
                size=(self._pos_e - self._pos_s,)
            ).astype(np.float32)

        # ── Wind latency: replace velocity obs with the reading from
        # wind_latency_steps steps ago.  A deque of fixed length is used
        # so that appending the current value automatically evicts the
        # oldest; the oldest (leftmost) entry is what the agent sees.
        if self.wind_latency_steps > 0:
            stale_vel = self._vel_buf[0].copy()           # oldest reading
            self._vel_buf.append(obs[self._vel_s : self._vel_e].copy())
            obs[self._vel_s : self._vel_e] = stale_vel

        # ── Packet loss: zero infection-level obs with probability p ─
        # The shared field map (infection levels) is transmitted via
        # inter-robot communication; a dropped packet means the robot
        # receives no update and sees zeros for that timestep.
        if self.packet_loss_prob > 0.0:
            if np.random.rand() < self.packet_loss_prob:
                obs[self._inf_s : self._inf_e] = 0.0

        return obs


# ────────────────────────────────────────────────────────────────
# Import the training MultiRobotEnv (env.py — supports dr_mode etc.)
# ────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join('.', 'src'))   # adjust if src/ is at a different path
from env import MultiRobotEnv


# ────────────────────────────────────────────────────────────────
# CoppeliaSim drone simulator (unchanged from your original)
# ────────────────────────────────────────────────────────────────
class DroneSimulator:
    def __init__(self, gym_env, sim, scaling_factor=8, height=0.4, num_robots=3):
        self.sim            = sim
        self.scaling_factor = scaling_factor
        self.height         = height
        self.num_robots     = num_robots
        self.polygon        = gym_env.unwrapped.poly_vertices
        scaled              = [(x / scaling_factor, y / scaling_factor)
                                for (x, y) in self.polygon]
        self.rounded_polygon = scaled + [scaled[0]]
        self.weed_locations  = list(gym_env.unwrapped.init_infected_positions)
        self.nozzle_scripts  = []
        self.all_drones      = []
        self.initial_positions = {}
        self.spawned_weeds   = []
        self.field_drawing   = None

        for i in range(num_robots):
            h = sim.getObject(f"/Quadcopter[{i}]/PaintNozzle/Script")
            self.nozzle_scripts.append(h)

        i = 0
        while True:
            h = sim.getObject(f"/Quadcopter[{i}]", {'noError': True})
            if h == -1:
                break
            self.all_drones.append(h)
            self.initial_positions[h] = sim.getObjectPosition(h, -1)
            i += 1

    def start(self):
        self.sim.startSimulation()

    def stop(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.step()
        self._stop_spraying()
        self._clear_field()
        self._clear_weeds()
        for drone, pos in self.initial_positions.items():
            self.sim.setObjectPosition(drone, -1, pos)
            self.sim.setModelProperty(drone, 0)

    def draw_field(self):
        self.field_drawing = self.sim.addDrawingObject(
            self.sim.drawing_lines, 5, 0, -1, 9999, [255, 255, 255])
        for i in range(len(self.rounded_polygon) - 1):
            p1, p2 = self.rounded_polygon[i], self.rounded_polygon[i + 1]
            self.sim.addDrawingObjectItem(
                self.field_drawing,
                [p1[0], p1[1], 0.1, p2[0], p2[1], 0.1])

    def set_agent_positions(self, info):
        for i, drone in enumerate(self.all_drones):
            if i < self.num_robots:
                pos = info[f'robot{i}']['position']
                self.sim.setObjectPosition(
                    drone, -1,
                    [p / self.scaling_factor for p in pos] + [self.height])
                self.sim.setObjectInt32Param(
                    drone, self.sim.objintparam_visibility_layer, 1)
            else:
                self.sim.setModelProperty(
                    drone,
                    self.sim.modelproperty_not_visible        |
                    self.sim.modelproperty_not_collidable     |
                    self.sim.modelproperty_not_detectable     |
                    self.sim.modelproperty_not_dynamic)

    def set_weed_locations(self):
        self._clear_weeds()
        weed_template = self.sim.getObject('/weed')
        for loc in self.weed_locations:
            new_pos  = [xi / self.scaling_factor for xi in loc] + [0]
            new_weed = self.sim.copyPasteObjects([weed_template])[0]
            self.sim.setObjectPosition(new_weed, -1, new_pos)
            self.spawned_weeds.append(new_weed)

    def move_agents(self, info, action):
        for i in range(self.num_robots):
            target = self.sim.getObject(f"/target[{i}]")
            pos    = info[f'robot{i}']['position']
            self.sim.setObjectPosition(
                target, -1,
                [p / self.scaling_factor for p in pos] + [self.height])
            spray  = float(action[i][2])
            self.sim.callScriptFunction(
                "setSprayCommand", self.nozzle_scripts[i], spray > 0.01, spray)

    def _stop_spraying(self):
        for i in range(self.num_robots):
            self.sim.callScriptFunction(
                "setSprayCommand", self.nozzle_scripts[i], True, 0)

    def _clear_field(self):
        if self.field_drawing is not None:
            self.sim.removeDrawingObject(self.field_drawing)
            self.field_drawing = None

    def _clear_weeds(self):
        for obj in self.spawned_weeds:
            if self.sim.isHandle(obj):
                self.sim.removeObject(obj)
        self.spawned_weeds = []


# ────────────────────────────────────────────────────────────────
# Single-episode runner
# ────────────────────────────────────────────────────────────────
def run_episode(env, model, perturber, drone_sim=None):
    obs, info = env.reset()
    perturber.reset()

    if drone_sim:
        drone_sim.set_agent_positions(info)
        drone_sim.set_weed_locations()
        drone_sim.start()

    total_reward = 0.0
    while True:
        perturbed_obs        = perturber.apply(obs)
        action, _            = model.predict(perturbed_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward        += reward

        if drone_sim:
            drone_sim.move_agents(info, action)
            # pygame.event.get()
            # pygame.time.wait(50)


        if terminated or truncated:
            break

    if drone_sim:
        drone_sim.stop()

    return total_reward


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main():
    json_dict  = load_experiment_dict_json(JSON_PATH)
    field_info = json_dict[f"set{ENV_VARIATION}"]

    N = NUM_ROBOTS
    M = len(field_info['infected_locations'])

    model = CrossQ.load(TRAINED_MODEL_PATH)
    print(f"Loaded model from {TRAINED_MODEL_PATH}")

    # ── Register env ──────────────────────────────────────────────
    env_id = 'MultiRobotEnvSim2Real-v0'
    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=MultiRobotEnv,
                     max_episode_steps=1000)

    env_kwargs = dict(
        field_info       = field_info,
        num_robots       = N,
        max_steps        = 1000,
        render_mode      = 'human' if RENDER_COPPELIA else None,
        obs_mode         = 'full',        # 5N+M — must match training
        dr_mode          = 'none',        # must match training (no DR)
        uncertainty_mode = 'full',
        reward_ablation  = 'full',
    )
    env = gym.make(env_id, **env_kwargs)

    # Sanity check: confirm obs dimension matches expected 5N+M
    expected_obs_dim = 5 * N + M
    actual_obs_dim   = env.observation_space.shape[0]
    assert actual_obs_dim == expected_obs_dim, (
        f"Obs dim mismatch: expected {expected_obs_dim} (5N+M = 5×{N}+{M}), "
        f"got {actual_obs_dim}. Check obs_mode='full' is active in env.py.")
    print(f"obs_dim = {actual_obs_dim}  (5×{N} + {M})  ✓")

    # ── Optional CoppeliaSim connection ───────────────────────────
    drone_sim = None
    if RENDER_COPPELIA:
        client    = RemoteAPIClient()
        sim_obj   = client.getObject('sim')
        sim_obj.setInt32Param(sim_obj.intparam_idle_fps, 0)
        drone_sim = DroneSimulator(
            env, sim_obj, scaling_factor=8, height=HEIGHT, num_robots=N)
        drone_sim.draw_field()

    # ── Perturbation conditions ────────────────────────────────────
    # (label, gps_noise, wind_latency_steps, packet_loss_prob)
    conditions = [
        ("No perturbation (baseline)",     0.0, 0, 0.00),
        ("GPS noise (σ=5.0 m)",            5.0, 0, 0.00),
        ("Wind estimate latency (3 step)", 0.0, 3, 0.00),
        ("Comm. packet loss (10%)",        0.0, 0, 0.10),
        ("All combined",                   5.0, 5, 0.10),
    ]

    results = {}
    for label, gps, wind_lat, pkt in conditions:
        print(f"\n{'='*60}")
        print(f"Condition : {label}")
        print(f"  gps_noise={gps}  wind_latency_steps={wind_lat}  packet_loss={pkt}")
        print(f"  Running {N_EVAL_EPISODES} episodes …")

        perturber      = ObsPerturber(N=N, M=M,
                                      gps_noise=gps,
                                      wind_latency_steps=wind_lat,
                                      packet_loss_prob=pkt)
        episode_rewards = []
        for ep in range(N_EVAL_EPISODES):
            r = run_episode(env, model, perturber, drone_sim=drone_sim)
            episode_rewards.append(r)
            print(f"  ep {ep+1:3d}/{N_EVAL_EPISODES}  reward={r:.2f}")

        iqm = compute_iqm(episode_rewards)
        results[label] = iqm
        print(f"  → IQM = {iqm:.2f}")

    # ── Print Table 8 ─────────────────────────────────────────────
    baseline = results["No perturbation (baseline)"]
    alone_map = {
        "No perturbation (baseline)":     "---",
        "GPS noise (σ=5.0 m)":            "✓",
        "Wind estimate latency (3 step)": "✓",
        "Comm. packet loss (10%)":        "✓",
        "All combined":                   "---",
    }

    lines = []
    lines.append("=" * 60)
    lines.append("TABLE 8 — copy these values into tab:obs_gap in full_experiments.tex")
    lines.append("=" * 60)
    lines.append(f"  {'Condition':<42} {'IQM':>8}  {'ΔIQM (%)':>10}  Alone")
    lines.append(f"  {'-'*42} {'-'*8}  {'-'*10}  -----")
    for label, iqm in results.items():
        delta = ((iqm - baseline) / abs(baseline) * 100) if baseline != 0 else 0.0
        sign  = "+" if delta > 0 else ""
        lines.append(f"  {label:<42} {iqm:>8.2f}  {sign}{delta:>8.1f}%   {alone_map[label]}")

    output = "\n".join(lines)
    print(f"\n{output}")

    with open("sim2real.out", "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print("\nResults saved to sim2real.out")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()