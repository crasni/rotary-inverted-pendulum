"""Learned correction from the rig's own logs, added to the simulated pendulum.

The rigid-body simulator mispredicts the real pendulum by about 2 degrees over
40 ms near upright, mostly in how recent commands and pendulum speed act. A
small network maps [sin, cos, pendulum speed, arm speed, arm, last four
commands] to an extra pendulum acceleration. `ResidualPendulum` applies it as a
generalized force during each 20 ms step.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

FEATURES = 9


def features(theta, pen_vel, arm_vel, arm, actions):
    """actions: newest first [a_k, a_k-1, a_k-2, a_k-3]."""
    return np.array([np.sin(theta), np.cos(theta), pen_vel / 10, arm_vel / 5, arm, *actions],
                    dtype=np.float32)


def network(hidden=64):
    return torch.nn.Sequential(torch.nn.Linear(FEATURES, hidden), torch.nn.Tanh(),
                               torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                               torch.nn.Linear(hidden, 1))


class ResidualModel:
    def __init__(self, path):
        blob = torch.load(Path(path), weights_only=False)
        self.net = network(blob["hidden"])
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.scale = float(blob["accel_scale"])

    def accel(self, x):
        with torch.no_grad():
            return float(self.net(torch.from_numpy(x[None]))[0, 0]) * self.scale


def upright_weight(theta, full=0.6, zero=1.0):
    """1 within `full` rad of upright, fading linearly to 0 at `zero` rad.

    The correction was fitted mostly on aggressive commands. Away from upright
    the plain simulator already matches gentle swing-ups on the rig, and an
    ungated correction invented energy there that a policy learned to exploit.
    """
    d = abs(float(np.remainder(theta + np.pi, 2 * np.pi) - np.pi))
    return float(np.clip((zero - d) / (zero - full), 0.0, 1.0))


class ResidualPendulum(gym.Wrapper):
    """Add the learned pendulum acceleration near upright. `gain` scales it (1 = as fitted)."""

    def __init__(self, env, model_path, gain=1.0, gate=True):
        super().__init__(env)
        self.model = ResidualModel(model_path)
        self.gain = gain
        self.gate = gate
        self.actions = [0.0] * 4

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.actions = [float(v) for v in obs[5::6][::-1]][:4]  # newest first (previous actions)
        return obs, info

    def step(self, action):
        plant = self.env.unwrapped
        a = float(np.clip(np.asarray(action).item(), -1.0, 1.0))
        self.actions = [a] + self.actions[:3]
        theta = float(plant.data.qpos[plant._pen_qpos_addr]) - np.pi
        x = features(theta, float(plant.data.qvel[plant._pen_qvel_addr]),
                     float(plant.data.qvel[plant._motor_qvel_addr]),
                     float(plant.data.qpos[plant._motor_qpos_addr]), self.actions)
        weight = upright_weight(theta) if self.gate else 1.0
        extra = 0.0 if weight == 0.0 else (
            weight * self.gain * self.model.accel(x) * plant.params.pendulum_inertia_kg_m2)
        addr = plant._pen_qvel_addr
        base = float(plant.data.qfrc_applied[addr])
        plant.data.qfrc_applied[addr] = base + extra
        try:
            return self.env.step(action)
        finally:
            plant.data.qfrc_applied[addr] = base
