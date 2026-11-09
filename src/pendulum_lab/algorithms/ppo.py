"""PPO training and deterministic deployment using Stable-Baselines3."""
from pathlib import Path
import json
import numpy as np

try:  # The rl extra provides torch and Stable-Baselines3; a minimal install may not.
    import torch
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:
    BaseFeaturesExtractor = None

if BaseFeaturesExtractor is not None:
    class ScaledObservations(BaseFeaturesExtractor):
        """Fixed SI-unit scaling stored inside the policy checkpoint.

        Defined at module level so checkpoints pickle it by import path. A class
        defined inside train() is pickled as bytecode, which only loads on the
        exact Python version that saved it (other versions segfault).
        """
        def __init__(self, observation_space):
            super().__init__(observation_space, features_dim=24)
            self.register_buffer("scale", torch.tensor([1.75, 1, 1, 5, 15, 1] * 4))

        def forward(self, observations):
            return observations / self.scale


class Controller:
    def __init__(self, *, params_path, config, checkpoint=None):
        if checkpoint is None:
            raise ValueError("PPO requires a trained --checkpoint; train in simulation first")
        from stable_baselines3 import PPO
        self.policy = PPO.load(str(checkpoint), device=config.get("device", "cpu"))

    def reset(self):
        pass

    def act(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (24,) or not np.isfinite(observation).all():
            raise ValueError("Expected a finite (24,) observation")
        action, _ = self.policy.predict(observation, deterministic=True)
        return float(np.clip(np.asarray(action).item(), -1, 1))


def train(*, env_factory, config, output_dir):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = int(config.get("total_steps", 100000))
    if steps <= 0:
        raise ValueError("total_steps must be positive")
    algorithm = PPO
    if config.get("cpu_rollouts", False) and str(config.get("device", "cpu")).startswith("cuda"):
        import torch

        class HybridPPO(PPO):
            """Collect small-MLP actions on CPU; optimize batches on CUDA."""
            def collect_rollouts(self, *args, **kwargs):
                training_device = self.device
                self.device = torch.device("cpu")
                self.policy.to(self.device)
                try:
                    return super().collect_rollouts(*args, **kwargs)
                finally:
                    self.device = training_device
                    self.policy.to(training_device)

        algorithm = HybridPPO
    options = dict(config.get("ppo", {}))
    if config.get("normalize_observations", False):
        policy_kwargs = dict(options.get("policy_kwargs", {}))
        if "features_extractor_class" in policy_kwargs:
            raise ValueError("normalize_observations cannot override a custom feature extractor")
        policy_kwargs["features_extractor_class"] = ScaledObservations
        options["policy_kwargs"] = policy_kwargs
    env = env_factory()
    try:
        reward_scale = float(config.get("reward_scale", 1.0))
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        if reward_scale != 1.0:
            from gymnasium.wrappers import TransformReward
            env = TransformReward(env, lambda reward: reward * reward_scale)
        env = Monitor(env, str(output_dir / "learning_curve"))
        policy = algorithm("MlpPolicy", env, seed=int(config.get("seed", 0)),
                       learning_rate=config.get("learning_rate", 0.0003),
                       device=config.get("device", "cpu"), verbose=1, **options)
        callback = CheckpointCallback(save_freq=max(1, int(config.get("checkpoint_every", 25000))),
                                      save_path=str(output_dir / "checkpoints"), name_prefix="ppo")
        policy.learn(total_timesteps=steps, callback=callback)
        policy.save(str(output_dir / "policy.zip"))
        (output_dir / "training.json").write_text(json.dumps(config, indent=2) + "\n")
    finally:
        env.close()
