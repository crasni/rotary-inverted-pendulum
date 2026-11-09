import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rig_curriculum as rc  # noqa: E402


def episode(path, segments, hz=50):
    """Write an episode whose pendulum is upright for each (start_s, length_s, arm_vel) segment."""
    n = 15 * hz
    x = np.zeros((n, 6))
    x[:, 0] = np.arange(n) / hz
    x[:, 2] = math.pi  # hanging
    for start, length, arm_vel in segments:
        i, j = int(start * hz), int((start + length) * hz)
        x[i:j, 2] = 0.0
        x[i:j, 3] = arm_vel
    np.savez(path, samples=x)


class CatchStatsTest(unittest.TestCase):
    def test_counts_early_holds_and_entry_direction(self):
        with tempfile.TemporaryDirectory() as d:
            episode(Path(d) / "episode_000.npz", [(2.0, 3.0, -1.0), (8.0, 0.5, 1.0), (10.0, 0.1, 1.0)])
            episode(Path(d) / "episode_001.npz", [(3.0, 0.6, 1.0), (6.0, 2.5, 1.0)])
            s = rc.catch_stats(d)
        self.assertEqual(s["catches"], 4)  # the 0.1 s blip is not a catch
        self.assertEqual((s["early"], s["early_held_2s"]), (2, 1))
        self.assertEqual(s["arm_neg"], [1, 0])
        self.assertEqual(s["arm_pos"], [1, 2])


class MirrorTest(unittest.TestCase):
    def test_mirror_frame_keeps_upright_and_flips_tilt(self):
        frame = np.array([0.2, math.sin(0.1), math.cos(0.1), 1.0, -2.0, 0.5])
        mirrored = frame * np.array(rc.MIRROR_FRAME)
        self.assertAlmostEqual(math.atan2(mirrored[1], mirrored[2]), -0.1)
        np.testing.assert_allclose(mirrored[[0, 3, 4, 5]], -frame[[0, 3, 4, 5]])


class ArmGuardPenaltyTest(unittest.TestCase):
    def test_off_by_default_and_grows_past_start_near_upright(self):
        from pendulum_lab.plant.reward import RewardWeights, arm_guard_penalty
        self.assertEqual(arm_guard_penalty(0.0, 1.3, RewardWeights()), 0.0)
        w = RewardWeights(k_arm_guard=20.0)
        self.assertEqual(arm_guard_penalty(0.0, math.radians(40), w), 0.0)
        near, far = arm_guard_penalty(0.0, math.radians(60), w), arm_guard_penalty(0.0, math.radians(77), w)
        self.assertGreater(far, near)
        self.assertAlmostEqual(far, arm_guard_penalty(0.0, -math.radians(77), w))
        self.assertLess(arm_guard_penalty(math.radians(90), math.radians(77), w), 0.01 * far)


if __name__ == "__main__":
    unittest.main()
