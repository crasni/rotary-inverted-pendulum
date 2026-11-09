import unittest
import numpy as np
from pendulum_lab.randomized_training import RandomizedPhysics, RANGES
from pendulum_lab.contracts import PARAMS
from pendulum_lab.plant.pendulum_env import PendulumParams


class RandomizedTrainingTests(unittest.TestCase):
    def test_seeded_resets_change_physics_and_preserve_contract(self):
        env = RandomizedPhysics()
        original = PARAMS.read_bytes()
        nominal = PendulumParams.load(PARAMS)
        try:
            a, info_a = env.reset(seed=73)
            b, info_b = env.reset(seed=73)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(info_a['randomized_physics'], info_b['randomized_physics'])
            self.assertEqual(a.shape, (24,))
            _, _, _, _, step_info = env.step(np.array([0.2], dtype=np.float32))
            case = info_a['randomized_physics']
            changed = env.env.unwrapped.params
            self.assertAlmostEqual(changed.pendulum_inertia_kg_m2,
                                   nominal.pendulum_inertia_kg_m2 * case['inertia_factor'])
            self.assertAlmostEqual(changed.pendulum_friction,
                                   nominal.pendulum_friction * case['friction_factor'])
            self.assertEqual(step_info['transport_delay_s'], case['delay_s'])
            self.assertEqual(step_info['motor_max_accel_rad_s2'], case['max_accel'])
            _, info_c = env.reset()
            self.assertNotEqual(case, info_c['randomized_physics'])
            for key, (low, high) in RANGES.items():
                self.assertTrue(low <= info_c['randomized_physics'][key] <= high)
        finally:
            env.close()
        self.assertEqual(PARAMS.read_bytes(), original)
