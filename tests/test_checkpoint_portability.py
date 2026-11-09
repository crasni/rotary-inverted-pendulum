"""Checkpoints must load on any supported Python, not only the one that saved them."""
import base64
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from pendulum_lab.contracts import PARAMS, observation_frame
from pendulum_lab.environment import make_env

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "ppo": ["runs/gpu_rl_20261016/ppo_hybrid_train/policy.zip"],
    "dqn": ["runs/gpu_rl_20261016/dqn_train/policy.zip"],
    "sac": ["runs/gpu_rl_20261016/sac_train/policy.zip",
            "runs/sac_robustness_20261016/train/policy.zip",
            "runs/sac_robustness_20261016/train/checkpoints/sac_350000_steps.zip",
            "runs/sac_robustness_20261016/train/checkpoints/sac_450000_steps.zip"],
}


def pickled_by_value(path):
    """True if policy_kwargs carries a class as bytecode (Python-version specific)."""
    with zipfile.ZipFile(path) as archive:
        kwargs = json.loads(archive.read("data"))["policy_kwargs"]
    return b"_make_skeleton_class" in base64.b64decode(kwargs.get(":serialized:", ""))


class CheckpointPortabilityTests(unittest.TestCase):
    def setUp(self):
        try:
            import stable_baselines3  # noqa: F401
        except ImportError:
            self.skipTest("install the rl optional dependencies")

    def test_committed_checkpoints_store_classes_by_import_path(self):
        # Checked before loading: a by-value class would segfault on other Pythons.
        for paths in CHECKPOINTS.values():
            for rel in paths:
                path = ROOT / rel
                if not path.exists():
                    continue
                with self.subTest(checkpoint=rel):
                    self.assertFalse(pickled_by_value(path))

    def test_committed_checkpoints_load_and_act(self):
        hanging = np.tile(observation_frame([0, np.pi, 0, 0], 0), 4)
        for name, paths in CHECKPOINTS.items():
            module = importlib.import_module("pendulum_lab.algorithms." + name)
            for rel in paths:
                path = ROOT / rel
                if not path.exists() or pickled_by_value(path):
                    continue
                with self.subTest(checkpoint=rel):
                    controller = module.Controller(params_path=PARAMS, config={}, checkpoint=path)
                    self.assertLessEqual(abs(controller.act(hanging)), 1)

    def test_scaled_training_saves_class_by_import_path(self):
        import torch
        torch.set_num_threads(1)
        for name in ("ppo", "sac", "dqn"):
            with self.subTest(algorithm=name), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module("pendulum_lab.algorithms." + name)
                settings = {"policy_kwargs": {"net_arch": [16, 16]}}
                if name == "ppo":
                    settings.update(n_steps=16, batch_size=8, n_epochs=1)
                else:
                    settings.update(learning_starts=4, buffer_size=100, batch_size=4,
                                    train_freq=1, gradient_steps=1)
                config = {"total_steps": 16, "seed": 3, "checkpoint_every": 16,
                          "normalize_observations": True, name: settings}
                module.train(env_factory=lambda: make_env(PARAMS, duration=0.2),
                             config=config, output_dir=Path(directory))
                self.assertFalse(pickled_by_value(Path(directory) / "policy.zip"))
