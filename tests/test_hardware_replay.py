"""Guard the controller that passed on hardware: replaying the recorded
passes must reproduce every command the rig was sent."""
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs/results/hardware/lqr"
PASSES = ["20261107_025738", "20261107_025856", "20261107_030202"]


class HardwarePassReplay(unittest.TestCase):
    def test_passing_runs_replay_exactly(self):
        try:
            import stable_baselines3  # noqa: F401  (SAC swing-up checkpoint)
        except ImportError:
            self.skipTest("RL dependencies are not installed")
        from pendulum_lab.replay import WARMUP_TICKS, replay

        for name in PASSES:
            with self.subTest(run=name):
                replayed, recorded, modes, recorded_modes = replay(RUNS / name, repo_root=ROOT)
                np.testing.assert_allclose(replayed[WARMUP_TICKS:], recorded[WARMUP_TICKS:],
                                           rtol=0, atol=1e-9)
                np.testing.assert_array_equal(modes, recorded_modes)


class LqrMatchPresets(unittest.TestCase):
    """PID and iLQR with the *_lqr_match presets reproduce the LQR pass.

    MPC's transfer preset passed on hardware because it sends LQR's commands.
    These presets give PID (a predictor in front of the PID loop) and iLQR (the
    same delay model and weights) the same property on the recorded pass.
    """

    def test_lqr_match_presets_replay_the_lqr_pass(self):
        try:
            import stable_baselines3  # noqa: F401  (SAC swing-up checkpoint)
        except ImportError:
            self.skipTest("RL dependencies are not installed")
        from pendulum_lab.replay import WARMUP_TICKS, replay

        run = RUNS / "20261107_030202"
        # 99th percentile of |command difference| during balance. iLQR keeps its
        # nonlinear terms, so it departs a little at larger angles.
        for algorithm, tolerance in (("mpc", 0.01), ("ilqr", 0.05), ("pid", 0.01)):
            name = "transfer" if algorithm == "mpc" else "lqr_match"
            with self.subTest(algorithm=algorithm):
                replayed, recorded, modes, recorded_modes = replay(
                    run, repo_root=ROOT, algorithm=algorithm,
                    config=f"configs/{algorithm}_my_rig_{name}.json")
                np.testing.assert_array_equal(modes[WARMUP_TICKS:], recorded_modes[WARMUP_TICKS:])
                balancing = recorded_modes > 0
                self.assertLess(np.percentile(np.abs(replayed - recorded)[balancing], 99), tolerance)


class HardwareIndexes(unittest.TestCase):
    def test_generated_indexes_are_up_to_date(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "promote_hardware_run", ROOT / "scripts/promote_hardware_run.py")
        promote = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(promote)
        self.assertEqual(promote.reindex(check=True), [],
                         "Run .venv/bin/python scripts/promote_hardware_run.py --reindex")


class TransferPresets(unittest.TestCase):
    """The MPC, iLQR and PID transfer presets reuse the passing LQR hand-off."""

    def test_transfer_presets_share_the_reference_handoff(self):
        import json
        from pendulum_lab.algorithms.lqr import HANDOFF_KEYS

        reference = json.loads((ROOT / "configs/lqr_my_rig_coupling_delay.json").read_text())
        lqr = reference["lqr"]
        limits = {"delay_s": lqr["delay"], "speed_rad_s": reference["arm_speed_target_rad_s"],
                  "travel_rad": reference["arm_travel_target_rad"]}
        for algorithm, name in (("mpc", "transfer"), ("ilqr", "transfer"), ("pid", "transfer"),
                                ("ilqr", "lqr_match"), ("pid", "lqr_match")):
            with self.subTest(preset=f"{algorithm}_my_rig_{name}"):
                preset = json.loads((ROOT / f"configs/{algorithm}_my_rig_{name}.json").read_text())
                self.assertEqual(preset["arm_command_limits"], limits)
                self.assertEqual({k: preset[algorithm].get(k) for k in HANDOFF_KEYS},
                                 {k: lqr.get(k) for k in HANDOFF_KEYS})


if __name__ == "__main__":
    unittest.main()
