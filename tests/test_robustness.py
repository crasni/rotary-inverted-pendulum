import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pendulum_lab.contracts import PARAMS
from pendulum_lab.environment import make_env, true_state
from pendulum_lab.plant.pendulum_env import PendulumParams
from pendulum_lab.robustness import ObservationNoise, stress_env, varied_parameters


class RobustnessTests(unittest.TestCase):
    def test_noise_is_seeded_and_history_is_coherent(self):
        env = ObservationNoise(make_env(PARAMS), 0.003, 0.1)
        try:
            first, _ = env.reset(seed=123)
            second, _ = env.reset(seed=123)
            np.testing.assert_array_equal(first, second)
            following, *_ = env.step(np.array([0.2]))
            np.testing.assert_array_equal(first[6:], following[:-6])
            self.assertEqual(following.shape, (24,))
            self.assertAlmostEqual(float(following[-5]**2 + following[-4]**2), 1, places=6)
            self.assertAlmostEqual(float(following[-1]), 0.2, places=6)
        finally:
            env.close()

    def test_zero_noise_preserves_observations_and_true_reward(self):
        base, noisy = make_env(PARAMS), ObservationNoise(make_env(PARAMS), 0, 0)
        try:
            a, _ = base.reset(seed=3)
            b, _ = noisy.reset(seed=3)
            np.testing.assert_array_equal(a, b)
            for action in [0.2, -0.1, 0]:
                a, reward_a, *_ = base.step(np.array([action]))
                b, reward_b, *_ = noisy.step(np.array([action]))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(reward_a, reward_b)
        finally:
            base.close()
            noisy.close()

    def test_sensor_noise_does_not_change_physics_or_reward(self):
        base = make_env(PARAMS)
        noisy = ObservationNoise(make_env(PARAMS), 0.01, 0.3)
        try:
            base.reset(seed=123)
            noisy.reset(seed=123)
            for action in [0.2, 0.2, -0.1, 0]:
                obs_a, reward_a, *_ = base.step(np.array([action]))
                obs_b, reward_b, *_ = noisy.step(np.array([action]))
                np.testing.assert_array_equal(true_state(base), true_state(noisy))
                self.assertEqual(reward_a, reward_b)
                self.assertFalse(np.array_equal(obs_a, obs_b))
        finally:
            base.close()
            noisy.close()

    def test_stress_changes_are_applied_at_reset_without_changing_base_file(self):
        before = PARAMS.read_bytes()
        nominal = PendulumParams.load(PARAMS)
        with tempfile.TemporaryDirectory() as directory:
            path = varied_parameters(PARAMS, {'friction_factor': 1.5, 'inertia_factor': 0.8},
                                     Path(directory) / 'params.json')
            changed = PendulumParams.load(path)
            self.assertAlmostEqual(changed.pendulum_inertia_kg_m2, 0.8 * nominal.pendulum_inertia_kg_m2)
            self.assertAlmostEqual(changed.pendulum_friction, 1.5 * nominal.pendulum_friction)
            env = stress_env(path, {'delay_s': 0.03, 'max_accel': 40, 'speed_limit': 4})
            try:
                env.reset(seed=1)
                _, _, _, _, info = env.step(np.array([1.0]))
                self.assertEqual(info['transport_delay_s'], 0.03)
                self.assertEqual(info['motor_max_accel_rad_s2'], 40)
                self.assertEqual(info['motor_planner_speed_limit_rad_s'], 4)
            finally:
                env.close()
        self.assertEqual(before, PARAMS.read_bytes())
