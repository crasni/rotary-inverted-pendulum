"""Fine-tune a SAC policy in simulation to balance with smooth commands.

docs/sim-fit-upright.md: near upright the simulator predicts the rig well for
smooth command changes and badly for abrupt reversals, and SAC reverses about
twice as often as LQR on the rig. This trains SAC (V1's own training simulator,
reward only) with an extra penalty on the tick-to-tick command change near
upright,  k * exp(-theta^2 / sigma^2) * (a_t - a_{t-1})^2,  so that its balancing
stays in the regime the simulator models correctly. The swing-up anchor of
rig_curriculum.py keeps the parent's rig-proven swing-up (states >= --anchor-min-deg).
"""
import argparse
import glob
import hashlib
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

RIG_JITTER = {"angle_rad": math.radians(0.6), "pen_vel": 0.6}  # rig reading jitter, rms


def t_noise(rng, rms, size=None):
    """Heavy-tailed (Student-t, 3 dof) noise with the given rms."""
    return rng.standard_t(3, size) * rms / math.sqrt(3.0)


class RigReadingNoise(gym.Wrapper):
    """Heavy-tailed rig-like jitter on the pendulum angle and speed the policy reads
    (scale x the measured rig rms), coherent across the four-frame history."""

    def __init__(self, env, scale, seed=0):
        super().__init__(env)
        self.scale, self.rng, self.frames = scale, np.random.default_rng(seed), None

    def noisy(self, frame):
        frame = np.array(frame, dtype=np.float32)
        theta = math.atan2(frame[1], frame[2]) + t_noise(self.rng, self.scale * RIG_JITTER["angle_rad"])
        frame[1:3] = [math.sin(theta), math.cos(theta)]
        frame[4] += t_noise(self.rng, self.scale * RIG_JITTER["pen_vel"])
        return frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [self.noisy(obs[i * 6:(i + 1) * 6]) for i in range(len(obs) // 6)]
        return np.concatenate(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = self.frames[1:] + [self.noisy(obs[-6:])]
        return np.concatenate(self.frames), reward, terminated, truncated, info


def add_observation_smoothness(model, weight, scale, seed, batch=256):
    """CAPS-style spatial smoothness: after each SAC training call, actor steps on
    weight * |mu(s) - mu(s~)|^2, s~ = s with rig-like jitter on every frame's pendulum
    angle and speed, so small reading errors cannot swing the command."""
    import torch
    rng = np.random.default_rng(seed)
    original = model.train

    def train(gradient_steps, batch_size=64):
        original(gradient_steps=gradient_steps, batch_size=batch_size)
        model.actor.set_training_mode(True)
        losses = []
        for _ in range(gradient_steps):
            obs = model.replay_buffer.sample(batch).observations
            o = obs.detach().cpu().numpy().copy()
            for f in range(o.shape[1] // 6):
                theta = np.arctan2(o[:, f * 6 + 1], o[:, f * 6 + 2]) + t_noise(rng, scale * RIG_JITTER["angle_rad"], len(o))
                o[:, f * 6 + 1], o[:, f * 6 + 2] = np.sin(theta), np.cos(theta)
                o[:, f * 6 + 4] += t_noise(rng, scale * RIG_JITTER["pen_vel"], len(o))
            a = model.actor(obs, deterministic=True)
            b = model.actor(torch.as_tensor(o, dtype=obs.dtype), deterministic=True)
            loss = weight * torch.mean((a - b) ** 2)
            model.actor.optimizer.zero_grad()
            loss.backward()
            model.actor.optimizer.step()
            losses.append(loss.item())
        model.actor.set_training_mode(False)
        if losses:
            model.logger.record("train/smoothness_loss", float(np.mean(losses)))

    model.train = train


class SmoothNearUpright(gym.Wrapper):
    """Subtract k * gate(theta) * (sent action change)^2 (times the reward scale)."""

    def __init__(self, env, weight, sigma_deg, scale):
        super().__init__(env)
        self.weight, self.sigma, self.scale = weight, math.radians(sigma_deg), scale
        self.last = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        change = float(obs[-1] - self.last[-1])          # newest frame's last action = command just sent
        theta = math.atan2(obs[-5], obs[-4])
        reward -= self.scale * self.weight * math.exp(-theta ** 2 / self.sigma ** 2) * change ** 2
        self.last = obs
        return obs, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=0.0, help="Command-change penalty weight k (0 = off)")
    parser.add_argument("--rig-noise", type=float, default=0.0, help="Rig-like reading jitter in training, x measured rms")
    parser.add_argument("--caps-weight", type=float, default=0.0, help="Observation-smoothness loss weight (0 = off)")
    parser.add_argument("--caps-scale", type=float, default=1.0, help="Perturbation size for that loss, x rig jitter")
    parser.add_argument("--gamma", type=float, help="Discount (default: the parent's 0.99)")
    parser.add_argument("--sigma-deg", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-min-deg", type=float, default=45.0)
    parser.add_argument("--anchor-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import rig_curriculum as rc
    from finetune_real import REWARD_SCALE
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure

    torch.set_num_threads(2)
    rc.check_lineage(args.checkpoint)
    env = rc.sim_env(args.seed)
    if args.weight:
        env = SmoothNearUpright(env, args.weight, args.sigma_deg, REWARD_SCALE)
    if args.rig_noise:
        env = RigReadingNoise(env, args.rig_noise, args.seed)
    extra = {"gamma": args.gamma} if args.gamma else {}
    model = SAC.load(args.checkpoint, env=env, device="cpu", learning_rate=args.learning_rate,
                     custom_objects={"replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {}}, **extra)
    model.set_random_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    model.set_logger(configure(str(args.output / "log"), ["csv"]))
    # Anchor states: SAC's real swing-up states from the curriculum batches.
    data, _ = rc.load_batches(sorted(b for b in glob.glob("runs/curriculum/[be][0-9]*")
                                     if glob.glob(f"{b}/transitions_*.npz")))
    far = data["obs"][:, -4] < math.cos(math.radians(args.anchor_min_deg))
    rc.add_swingup_anchor(model, data["obs"][far], args.anchor_steps, args.seed)
    if args.caps_weight:
        add_observation_smoothness(model, args.caps_weight, args.caps_scale, args.seed)
    model.learning_starts = model.num_timesteps + 2000
    t0 = time.time()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    model.__dict__.pop("train", None)
    model.save(args.output / "policy.zip")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (args.output / "version.json").write_text(json.dumps({
        "method": "SAC fine-tuned in V1's training simulator, reward only; optional near-upright penalty on "
                  "the command change per tick, rig-like reading noise, observation-smoothness loss, discount; "
                  "swing-up anchored to the parent; no rig data, no imitation",
        "parent": str(args.checkpoint), "parent_sha256": sha(args.checkpoint), "weight": args.weight,
        "rig_noise": args.rig_noise, "caps_weight": args.caps_weight, "caps_scale": args.caps_scale,
        "gamma": float(model.gamma),
        "sigma_deg": args.sigma_deg, "steps": args.steps, "learning_rate": args.learning_rate,
        "anchor_min_deg": args.anchor_min_deg, "anchor_states": int(far.sum()), "seed": args.seed,
        "minutes": round((time.time() - t0) / 60, 1), "policy_sha256": sha(args.output / "policy.zip"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")
    print(f"saved {args.output / 'policy.zip'} after {args.steps} steps", flush=True)


if __name__ == "__main__":
    main()
