import unittest

import numpy as np

from pendulum_lab.disturbance import PendulumDisturbance
from pendulum_lab.environment import make_env
from pendulum_lab.contracts import PARAMS


class DisturbanceTests(unittest.TestCase):
    def test_seed_reproduction_and_unseeded_progress(self):
        env = PendulumDisturbance(make_env(PARAMS), randomize=True)
        try:
            def rollout():
                obs, info = env.reset(seed=73)
                values = []
                for _ in range(10):
                    obs, *_ = env.step(np.array([0.0], dtype=np.float32))
                    values.append(env.value)
                return obs, values, info['disturbance_rms']
            a, noise_a, rms = rollout()
            b, noise_b, repeated_rms = rollout()
            np.testing.assert_array_equal(a, b)
            self.assertEqual(noise_a, noise_b)
            self.assertEqual(rms, repeated_rms)
            _, info = env.reset()
            self.assertNotEqual(rms, info['disturbance_rms'])
            self.assertTrue(0 <= info['disturbance_rms'] <= 9)
            plant = env.unwrapped
            self.assertEqual(plant.data.qfrc_applied[plant._pen_qvel_addr], 0)
            env.step(np.array([0.0], dtype=np.float32))
            self.assertAlmostEqual(plant.data.qfrc_applied[plant._pen_qvel_addr],
                                   env.value * plant.params.pendulum_inertia_kg_m2)
        finally:
            env.close()

    def test_zero_disturbance_preserves_trajectory(self):
        plain = make_env(PARAMS)
        wrapped = PendulumDisturbance(make_env(PARAMS), accel_rms=0)
        try:
            a, _ = plain.reset(seed=42)
            b, _ = wrapped.reset(seed=42)
            np.testing.assert_array_equal(a, b)
            for action in [0.2, -0.4, 0.0, 0.8] * 5:
                a, reward_a, done_a, truncated_a, _ = plain.step(np.array([action], dtype=np.float32))
                b, reward_b, done_b, truncated_b, _ = wrapped.step(np.array([action], dtype=np.float32))
                np.testing.assert_array_equal(a, b)
                self.assertEqual((reward_a, done_a, truncated_a), (reward_b, done_b, truncated_b))
        finally:
            plain.close()
            wrapped.close()

    def test_invalid_noise_settings(self):
        env = make_env(PARAMS)
        try:
            for rms, correlation in [(-1, .3), (9, 1), (9, -1), (float('nan'), .3)]:
                with self.assertRaises(ValueError):
                    PendulumDisturbance(env, rms, correlation)
        finally:
            env.close()
