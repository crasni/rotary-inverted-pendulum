"""Fine-tune a saved RL policy on the identified rig in simulation."""
import argparse
import dataclasses
import json
import hashlib
from pathlib import Path
import shutil

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from pendulum_lab.disturbance import PendulumDisturbance
from pendulum_lab.arm_limits import limit_arm_action
from pendulum_lab.contracts import state_from_obs
from pendulum_lab.residual import ResidualPendulum
from pendulum_lab.robustness import stress_env

BASELINES = {
    "ppo": (PPO, "runs/gpu_rl_20261016/ppo_hybrid_train/policy.zip", 0.05),
    "dqn": (DQN, "runs/gpu_rl_20261016/dqn_train/policy.zip", 0.1),
    "sac": (SAC, "docs/results/sac_randomized_policy.zip", 0.2),
}
RIG = {"delay_s": 0.018, "speed_limit": 4.4}


class RandomDelay(gym.Wrapper):
    """Sample the command delay, and optionally the pendulum reading lag, per episode."""

    def __init__(self, env, low, high, pen_lag=(0.0, 0.0)):
        super().__init__(env)
        self.low, self.high = low, high
        self.pen_lag = pen_lag
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 11)
        plant = self.env.unwrapped
        plant._fixed_transport_delay_s = float(self.rng.uniform(self.low, self.high))
        plant._pen_obs_lag_s = float(self.rng.uniform(*self.pen_lag))
        return self.env.reset(seed=seed, options=options)


class RandomPhysics(gym.Wrapper):
    """Per episode, scale gravity (swing frequency), pendulum damping and arm acceleration.

    A swing-up tuned to one exact model arrives at the top too fast on the rig,
    because a few percent of energy error is several rad/s there. Varying these
    forces the policy to watch the pendulum energy instead.
    The policy still sees the commands it sent, not the scaled ones.
    """

    RANGES = {"gravity": (0.9, 1.1), "damping": (0.5, 4.0), "arm_gain": (0.9, 1.15)}

    def __init__(self, env):
        super().__init__(env)
        self.rng = np.random.default_rng()
        self.gain = 1.0
        self.sent = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 31)
        obs, info = self.env.reset(seed=seed, options=options)
        plant = self.env.unwrapped
        g, d, self.gain = (float(self.rng.uniform(*self.RANGES[k])) for k in ("gravity", "damping", "arm_gain"))
        plant.model.opt.gravity[:] *= g
        plant.model.dof_damping[plant._pen_dof_addr] *= d
        self.sent = [float(v) for v in obs[5::6]]
        return obs, {**info, "physics": {"gravity": g, "damping": d, "arm_gain": self.gain}}

    def step(self, action):
        sent = float(np.asarray(action).item())
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([np.clip(sent * self.gain, -1.0, 1.0)], dtype=np.float32))
        self.sent = self.sent[1:] + [sent]
        obs = obs.copy()
        obs[5::6] = self.sent
        return obs, reward, terminated, truncated, info


class RandomArmLag(gym.Wrapper):
    """Sample the physical arm's lag behind the step count per episode. Braking a
    fast-spinning pendulum is timing-sensitive (a few ms shift the push by tens of
    degrees at 5 rev/s), and the rig's timing differs from the nominal model there."""

    def __init__(self, env, low, high, coupling=(1.0, 1.0)):
        super().__init__(env)
        self.low, self.high, self.coupling = low, high, coupling
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 53)
        plant = self.env.unwrapped
        plant._arm_lag_s = float(self.rng.uniform(self.low, self.high))
        # Fraction of the modelled arm-to-pendulum coupling that acts. Rig spin
        # segments (docs/sim-fit-upright.md) fit 1.2-1.4: the real arm pushes harder.
        plant._coupling_gain = float(self.rng.uniform(*self.coupling))
        return self.env.reset(seed=seed, options=options)


class EnergyCap(gym.Wrapper):
    """Penalise pendulum energy above what reaching the top needs.

    Normalised energy e = 0.5 * (pen_vel / omega)^2 + cos(theta), theta from
    upright: 1 = at rest upright, -1 = at rest hanging. Swing-ups and catches arrive
    with e of about 1.0-1.1; a pendulum spinning at 8 rad/s has e of about 1.2. The penalty
    weight * max(e - threshold, 0) makes spinning (which otherwise earns more than
    hanging) the worse choice, so braking and re-catching pays off.
    """

    def __init__(self, env, weight, threshold=1.1, omega=12.3):
        super().__init__(env)
        self.weight, self.threshold, self.omega = weight, threshold, omega

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        energy = 0.5 * (float(obs[-2]) / self.omega) ** 2 + float(obs[-4])  # newest frame: cos(theta), pen_vel
        reward -= self.weight * max(energy - self.threshold, 0.0)
        return obs, reward, terminated, truncated, info


class RandomResidual(ResidualPendulum):
    """The rig-fitted pendulum correction, with its strength sampled per episode."""

    def __init__(self, env, model_path, low, high):
        super().__init__(env, model_path)
        self.low, self.high = low, high
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 41)
        self.gain = float(self.rng.uniform(self.low, self.high))
        return super().reset(seed=seed, options=options)


class LimitedActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.limits = json.loads(Path("configs/sac_my_rig.json").read_text())["arm_command_limits"]

    def reset(self, **kwargs):
        self.observation, info = self.env.reset(**kwargs)
        return self.observation, info

    def step(self, action):
        arm, _, speed, _ = state_from_obs(self.observation)
        limited = limit_arm_action(
            float(np.asarray(action).item()), arm, speed, self.observation[-1],
            delay=self.limits["delay_s"], speed_target=self.limits["speed_rad_s"],
            travel_target=self.limits["travel_rad"],
        )
        self.observation, reward, terminated, truncated, info = self.env.step(
            np.array([limited], dtype=np.float32)
        )
        return self.observation, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=BASELINES)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Continue an already adapted policy")
    parser.add_argument("--disturbance-rms", type=float, default=0.0,
                        help="Sample episode acceleration RMS uniformly from zero to this value")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--reset-bank", type=Path,
                        help="Observations measured on the rig (scripts/real_state_starts.py bank)")
    parser.add_argument("--bank-frac", type=float, default=0.4,
                        help="Share of episodes that start from the reset bank")
    parser.add_argument("--pen-lag-range", type=float, nargs=2, default=(0.0, 0.0), metavar=("LOW", "HIGH"),
                        help="Extra age of the pendulum reading, sampled per episode, in seconds")
    parser.add_argument("--fast-arrival-frac", type=float, default=0.0,
                        help="Share of episodes that start near upright with the pendulum moving 2-9 rad/s")
    parser.add_argument("--residual", type=Path,
                        help="Rig-fitted pendulum correction (src/pendulum_lab/residual.py)")
    parser.add_argument("--residual-gain", type=float, nargs=2, default=(1.0, 1.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--random-physics", action="store_true",
                        help="Vary gravity, pendulum damping and arm acceleration per episode")
    parser.add_argument("--action-rate-weight", type=float, default=0.0,
                        help="Penalty on the squared change between consecutive commands")
    parser.add_argument("--fast-arrival-speed", type=float, nargs=2, default=(2.0, 9.0), metavar=("LOW", "HIGH"),
                        help="Pendulum speed range for --fast-arrival-frac starts (rad/s); 6-15 = already spinning")
    parser.add_argument("--arm-lag-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Sample the physical arm lag per episode, in seconds")
    parser.add_argument("--coupling-range", type=float, nargs=2, default=(1.0, 1.0), metavar=("LOW", "HIGH"),
                        help="With --arm-lag-range: sample the arm-to-pendulum coupling gain per episode")
    parser.add_argument("--energy-cap", type=float, default=0.0,
                        help="Penalty weight on pendulum energy above the top (stops the spinning exploit; 0 = off)")
    parser.add_argument("--energy-threshold", type=float, default=1.1)
    parser.add_argument("--rig-noise", type=float, default=0.0,
                        help="Rig-like heavy-tailed reading jitter on pendulum angle/speed, x measured rms "
                             "(train_smooth_sac.RigReadingNoise; what made SAC rnA pass on the rig)")
    parser.add_argument("--delay-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Sample the command delay per episode, in seconds")
    args = parser.parse_args()
    if not np.isfinite(args.disturbance_rms) or args.disturbance_rms < 0:
        parser.error("--disturbance-rms must be finite and nonnegative")
    if args.learning_rate is not None and (not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate must be finite and positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    algorithm, checkpoint, scale = BASELINES[args.algorithm]
    checkpoint = str(args.checkpoint or checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    env = stress_env(Path("configs/my_rig.json"), RIG)
    env.unwrapped.exact_hanging_reset_frac = 0.25
    env.unwrapped.upright_reset_frac = 0.25
    if args.reset_bank:
        plant = env.unwrapped
        plant._spin_reset_obs = np.load(args.reset_bank)["obs"].astype(np.float32)
        plant.spin_reset_frac = args.bank_frac
        rest = (1 - args.bank_frac) / 2
        plant.exact_hanging_reset_frac = rest * 0.6
        plant.upright_reset_frac = rest * 0.4
    if args.action_rate_weight:
        plant = env.unwrapped
        plant._reward_weights = dataclasses.replace(plant._reward_weights,
                                                    k_action_rate=args.action_rate_weight)
    if args.fast_arrival_frac:
        plant = env.unwrapped
        plant.recovery_reset_frac = args.fast_arrival_frac
        plant.recovery_theta_max_rad = 0.8
        plant.recovery_pen_vel_min_rad_s, plant.recovery_pen_vel_max_rad_s = args.fast_arrival_speed
        plant.recovery_motor_vel_max_rad_s = 3.0
        rest = 1 - plant.spin_reset_frac - plant.recovery_reset_frac
        plant.exact_hanging_reset_frac = rest * 0.5
        plant.upright_reset_frac = rest * 0.2
    if args.delay_range:
        env = RandomDelay(env, *args.delay_range, pen_lag=tuple(args.pen_lag_range))
    if args.random_physics:
        env = RandomPhysics(env)
    if args.residual:
        env = RandomResidual(env, args.residual, *args.residual_gain)
    if args.disturbance_rms:
        env = PendulumDisturbance(env, args.disturbance_rms, randomize=True)
    if args.arm_lag_range:
        env = RandomArmLag(env, *args.arm_lag_range, coupling=tuple(args.coupling_range))
    if args.energy_cap:
        env = EnergyCap(env, args.energy_cap, args.energy_threshold)
    if args.rig_noise:
        from train_smooth_sac import RigReadingNoise
        env = RigReadingNoise(env, args.rig_noise, args.seed)
    env = gym.wrappers.TransformReward(LimitedActions(env), lambda reward: reward * scale)
    sidecar = Path(checkpoint + ".actions.json")
    try:
        if args.algorithm == "dqn":
            from pendulum_lab.algorithms.dqn import DiscreteActions
            env = DiscreteActions(env, json.loads(sidecar.read_text()))
        overrides = {} if args.learning_rate is None else {"learning_rate": args.learning_rate}
        policy = algorithm.load(checkpoint, env=env, device="cpu", verbose=1, **overrides)
        policy.set_random_seed(args.seed)
        if args.algorithm != "ppo":
            policy.learning_starts = policy.num_timesteps + 5_000
        callback = CheckpointCallback(save_freq=25_000, save_path=str(args.output / "checkpoints"),
                                      name_prefix=args.algorithm)
        initial_steps = policy.num_timesteps
        policy.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callback)
        policy.save(args.output / "policy.zip")
        if args.algorithm == "dqn":
            shutil.copyfile(sidecar, args.output / "policy.zip.actions.json")
            for saved in (args.output / "checkpoints").glob("*.zip"):
                shutil.copyfile(sidecar, Path(str(saved) + ".actions.json"))
        (args.output / "training.json").write_text(json.dumps({
            "algorithm": args.algorithm, "seed": args.seed, "source": checkpoint,
            "source_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "disturbance_rms_range": [0, args.disturbance_rms], "correlation": 0.3,
            "learning_rate": policy.learning_rate, "requested_steps": args.steps,
            "initial_steps": initial_steps, "final_steps": policy.num_timesteps,
            "params": "configs/my_rig.json", "rig": RIG,
            "reset_bank": str(args.reset_bank) if args.reset_bank else None,
            "bank_frac": args.bank_frac if args.reset_bank else 0.0,
            "delay_range": args.delay_range,
            "action_rate_weight": args.action_rate_weight,
            "pen_lag_range": list(args.pen_lag_range),
            "fast_arrival_frac": args.fast_arrival_frac,
            "fast_arrival_speed": list(args.fast_arrival_speed),
            "residual": str(args.residual) if args.residual else None,
            "residual_gain": list(args.residual_gain),
            "random_physics": RandomPhysics.RANGES if args.random_physics else None,
            "rig_noise": args.rig_noise,
            "arm_lag_range": args.arm_lag_range,
            "coupling_range": list(args.coupling_range),
            "energy_cap": args.energy_cap, "energy_threshold": args.energy_threshold,
        }, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
