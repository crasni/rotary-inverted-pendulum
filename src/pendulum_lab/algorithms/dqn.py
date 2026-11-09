"""DQN training and deterministic deployment using Stable-Baselines3."""
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
import gymnasium as gym


class DiscreteActions(gym.ActionWrapper):
    """Save the action grid with the checkpoint so inference uses the same map."""
    def __init__(self, env, values):
        super().__init__(env)
        self.values = action_grid(values)
        self.action_space = gym.spaces.Discrete(len(self.values))

    def action(self, action):
        return np.array([self.values[int(action)]], dtype=np.float32)


def action_grid(values):
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or np.any(abs(grid) > 1):
        raise ValueError("DQN actions must be a finite list of at least two values in [-1, 1]")
    return grid


class Controller:
    def __init__(self, *, params_path, config, checkpoint=None):
        if checkpoint is None:
            raise ValueError("DQN requires a trained --checkpoint; train in simulation first")
        from stable_baselines3 import DQN
        self.policy = DQN.load(str(checkpoint), device=config.get("device", "cpu"))
        self.values = action_grid(json.loads(Path(str(checkpoint) + ".actions.json").read_text()))
        if self.policy.action_space.n != len(self.values):
            raise ValueError("DQN checkpoint and action grid differ")

    def reset(self):
        pass

    def act(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (24,) or not np.isfinite(observation).all():
            raise ValueError("Expected a finite (24,) observation")
        action, _ = self.policy.predict(observation, deterministic=True)
        return float(self.values[int(np.asarray(action).item())])


def train(*, env_factory, config, output_dir):
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = int(config.get("total_steps", 100000))
    if steps <= 0:
        raise ValueError("total_steps must be positive")
    options = dict(config.get("dqn", {}))
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
        values = action_grid(config.get("actions", [-1, -0.5, 0, 0.5, 1]))
        env = DiscreteActions(env, values)
        env = Monitor(env, str(output_dir / "learning_curve"))
        policy = DQN("MlpPolicy", env, seed=int(config.get("seed", 0)),
                       learning_rate=config.get("learning_rate", 0.0003),
                       device=config.get("device", "cpu"), verbose=1, **options)
        class ActionCheckpoint(CheckpointCallback):
            def _on_step(self):
                keep_training = super()._on_step()
                if self.n_calls % self.save_freq == 0:
                    path = Path(self.save_path) / f"{self.name_prefix}_{self.num_timesteps}_steps.zip.actions.json"
                    path.write_text(json.dumps(values.tolist()) + "\n")
                return keep_training

        callback = ActionCheckpoint(save_freq=max(1, int(config.get("checkpoint_every", 25000))),
                                      save_path=str(output_dir / "checkpoints"), name_prefix="dqn")
        policy.learn(total_timesteps=steps, callback=callback)
        policy.save(str(output_dir / "policy.zip"))
        (output_dir / "policy.zip.actions.json").write_text(json.dumps(values.tolist()) + "\n")
        (output_dir / "training.json").write_text(json.dumps(config, indent=2) + "\n")
    finally:
        env.close()
