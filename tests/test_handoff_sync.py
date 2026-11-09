"""The classical controllers carry identical copies of the shared hand-off block."""
import importlib.util
import inspect
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_sync_script():
    spec = importlib.util.spec_from_file_location("sync_handoff", ROOT / "scripts/sync_handoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HandoffSync(unittest.TestCase):
    def test_copies_match_lqr(self):
        sync = load_sync_script()
        canonical = sync.block(sync.SOURCE)
        for name in sync.COPIES:
            with self.subTest(file=name):
                self.assertEqual(sync.block(name), canonical,
                                 f"Run .venv/bin/python scripts/sync_handoff.py after editing lqr.py")

    def test_inline_arm_limiter_matches_shared_wrapper(self):
        from pendulum_lab import arm_limits
        from pendulum_lab.algorithms import lqr

        for name in ("limit_arm_action", "validate_arm_limits"):
            with self.subTest(function=name):
                self.assertEqual(inspect.getsource(getattr(lqr, name)),
                                 inspect.getsource(getattr(arm_limits, name)))


if __name__ == "__main__":
    unittest.main()
