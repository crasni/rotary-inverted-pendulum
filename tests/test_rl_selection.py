"""Selection must not trade a safety stop for more successful episodes."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'select_disturbed_rl', Path(__file__).resolve().parents[1] / 'scripts/select_disturbed_rl.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class SelectionTests(unittest.TestCase):
    def test_safety_then_success_then_balance(self):
        def candidate(stops, passes, balance):
            return {'cases': {'case': {'safety_stops': stops, 'passed': passes,
                                      'mean_longest_balance_s': balance}}}
        safe = candidate(0, 1, 3)
        unsafe = candidate(1, 20, 15)
        self.assertGreater(selection.rank(safe), selection.rank(unsafe))
        self.assertGreater(selection.rank(candidate(0, 2, 1)), selection.rank(safe))
        self.assertGreater(selection.rank(candidate(0, 1, 4)), selection.rank(safe))
