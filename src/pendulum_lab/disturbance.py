"""Seeded pendulum torque disturbance adapted from crasni's 17f629e experiment."""
import math

import gymnasium as gym
import numpy as np


class PendulumDisturbance(gym.Wrapper):
    """AR(1) acceleration noise, optionally sampling RMS per training episode.

    9 rad/s² RMS and correlation 0.3 reproduce the teammate's diagnostic.
    These are surrogate disturbances, not identified physical noise bounds.
    """

    def __init__(self, env, accel_rms=9.0, correlation=0.3, randomize=False):
        super().__init__(env)
        if not np.isfinite([accel_rms, correlation]).all() or accel_rms < 0 or abs(correlation) >= 1:
            raise ValueError("Require finite RMS >= 0 and abs(correlation) < 1")
        self.accel_rms = float(accel_rms)
        self.correlation = float(correlation)
        self.randomize = randomize
        self.rng = np.random.default_rng()
        self.value = 0.0
        self.episode_rms = self.accel_rms

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 7)
        self.value = 0.0
        self.episode_rms = self.rng.uniform(0, self.accel_rms) if self.randomize else self.accel_rms
        obs, info = self.env.reset(seed=seed, options=options)
        self.env.unwrapped.data.qfrc_applied[self.env.unwrapped._pen_qvel_addr] = 0.0
        return obs, {**info, "disturbance_rms": self.episode_rms}

    def step(self, action):
        plant = self.env.unwrapped
        self.value = (self.correlation * self.value + math.sqrt(1 - self.correlation**2)
                      * self.rng.normal() * self.episode_rms)
        plant.data.qfrc_applied[plant._pen_qvel_addr] = self.value * plant.params.pendulum_inertia_kg_m2
        return self.env.step(action)
