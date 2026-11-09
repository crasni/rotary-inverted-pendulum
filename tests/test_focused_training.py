import json
from pathlib import Path
import unittest

import numpy as np

from pendulum_lab.focused_training import FailureSampler, FocusedPhysics, validation_cases, select_score
from pendulum_lab.randomized_training import RANGES


class FocusedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.sampler = FailureSampler(json.loads(
            Path('docs/results/sac_randomized_20261022.json').read_text()))

    def test_failure_sampling_weights_mixture_and_bounds(self):
        self.assertEqual(len(self.sampler.failures), 9)
        self.assertAlmostEqual(self.sampler.weights.sum(), 1)
        self.assertAlmostEqual(self.sampler.weights[6], 10 / 38)
        rng = np.random.default_rng(174)
        counts = {'exact_failure': 0, 'near_failure': 0, 'broad': 0}
        for _ in range(4000):
            case = self.sampler.sample(rng)
            counts[case['sampling_source']] += 1
            for key, (low, high) in RANGES.items():
                self.assertTrue(low <= case[key] <= high)
        for name, expected in [('exact_failure', 0.25), ('near_failure', 0.40), ('broad', 0.35)]:
            self.assertAlmostEqual(counts[name] / 4000, expected, delta=0.025)

    def test_focused_environment_is_seeded_and_keeps_guards(self):
        env = FocusedPhysics(self.sampler)
        try:
            a, info_a = env.reset(seed=174)
            b, info_b = env.reset(seed=174)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(info_a['randomized_physics'], info_b['randomized_physics'])
            self.assertEqual(env.env.unwrapped.upright_reset_frac, 0.2)
            self.assertEqual(env.env.unwrapped.exact_hanging_reset_frac, 0.1)
            _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
            self.assertEqual(info['motor_max_accel_rad_s2'], info_a['randomized_physics']['max_accel'])
        finally:
            env.close()

    def test_validation_is_reproducible_and_selection_prefers_safety(self):
        self.assertEqual(validation_cases(self.sampler), validation_cases(self.sampler))
        self.assertEqual(len(validation_cases(self.sampler)), 12)
        unsafe = {'results': [{'safety_stops': 1, 'successes': 59, 'mean_longest_balance_s': 13}]}
        safe = {'results': [{'safety_stops': 0, 'successes': 50, 'mean_longest_balance_s': 11}]}
        self.assertGreater(select_score(safe), select_score(unsafe))
