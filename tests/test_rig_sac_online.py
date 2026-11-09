import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rig_sac_online as ro  # noqa: E402


class PinkNoiseTest(unittest.TestCase):
    def test_unit_variance_zero_mean_and_correlated(self):
        x = ro.pink_noise(1000, np.random.default_rng(1))
        self.assertAlmostEqual(x.std(), 1.0, places=6)
        self.assertAlmostEqual(x.mean(), 0.0, places=6)
        lag = np.corrcoef(x[:-10], x[10:])[0, 1]
        self.assertGreater(lag, 0.1)  # white noise would be ~0


class LayerNormCriticsTest(unittest.TestCase):
    def test_checkpoint_keeps_standard_layout_and_trained_actor(self):
        import torch
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", "Pendulum-v1", device="cpu", learning_starts=0, buffer_size=1000)
        model.learn(64)
        critics = ro.LayerNormCritics(model, [32, 32], 0.01, 1e-3)
        self.assertIsInstance(model.critic.q_networks[0][2], torch.nn.LayerNorm)
        model.train(gradient_steps=5, batch_size=32)
        actor = {k: v.clone() for k, v in model.actor.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            critics.save(Path(d) / "ck.zip")
            loaded = SAC.load(Path(d) / "ck.zip", device="cpu")  # what the runner does
            self.assertTrue((Path(d) / "ck.critic_ln.pt").exists())
        for k, v in loaded.actor.state_dict().items():
            self.assertTrue(torch.equal(v, actor[k]), k)
        self.assertIsInstance(model.critic.q_networks[0][2], torch.nn.LayerNorm)  # swapped back
        model.train(gradient_steps=2, batch_size=32)


if __name__ == "__main__":
    unittest.main()
