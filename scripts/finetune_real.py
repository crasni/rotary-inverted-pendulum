"""Fine-tune SAC on logged rig transitions mixed with simulator experience.

Follows the RLPD recipe (Ball et al., 2023): every critic/actor batch is half
logged hardware transitions and half fresh simulator transitions. The simulator
half keeps the swing-up, which the plain simulator models well; the hardware
half teaches the critic what actually happens near upright on this rig,
including from the classical controllers' successful balancing.

The policy still swings up and balances on its own: hardware data only enters
SAC's own off-policy updates. Every fifth logged run is held out and the critic's
TD error on it is logged next to the training runs' to detect overfitting.
"""
import argparse
import glob
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from pendulum_lab.contracts import observation_frame
from pendulum_lab.disturbance import PendulumDisturbance
from pendulum_lab.plant.reward import compute_reward
from pendulum_lab.robustness import ObservationNoise, stress_env
from train_rig_rl import RIG, LimitedActions, RandomDelay

REWARD_SCALE = 0.2  # train_rig_rl's SAC scale
RL = ("sac", "ppo", "dqn")
EXCLUDE = ("20261107_163412",)  # encoder stall, not policy behaviour


def default_runs():
    runs = glob.glob("docs/results/hardware/*/2026*") + glob.glob("runs/*_rig_2026*") \
        + glob.glob("runs/sac_[A-Z]_2026*")
    return sorted({r for r in runs if not any(x in r for x in EXCLUDE)})


def run_transitions(run, weights):
    """(obs, next_obs, action, reward, done) for one logged run."""
    run = Path(run)
    data = np.load(run / "episode_000.npz", allow_pickle=True)
    samples, timing = data["samples"], data["hardware_timing"]
    algorithm = json.loads((run / "run.json").read_text())["algorithm"]
    summary = json.loads((run / "summary.json").read_text())[0]
    states, sent = samples[:, 1:5], samples[:, 5]
    requested = timing[:, 7]
    n = len(samples)
    if n < 20:
        return None

    def history(k):
        return np.concatenate([observation_frame(states[j], sent[j - 1] if j > 0 else 0.0)
                               for j in range(k - 3, k + 1)])

    out = {key: [] for key in ("obs", "next_obs", "action", "reward", "done")}
    guard_stop = summary["stop_reason"] not in ("duration",) and not summary["stop_reason"].startswith("Keyboard")
    for k in range(4, n - 1):
        policy_action = requested[k] if algorithm in RL and np.isfinite(requested[k]) else sent[k]
        arm, theta, arm_vel, pen_vel = states[k + 1]
        reward = compute_reward(theta=theta, pen_vel=pen_vel, motor_pos=arm, motor_vel=arm_vel,
                                action=float(sent[k]), prev_action=float(sent[k - 1]),
                                prev_motor_vel=float(states[k, 2]), weights=weights)
        last = k == n - 2
        done = guard_stop and last
        if done:
            reward -= 100.0
        out["obs"].append(history(k))
        out["next_obs"].append(history(k + 1))
        out["action"].append([float(np.clip(policy_action, -1, 1))])
        out["reward"].append(reward * REWARD_SCALE)
        out["done"].append(float(done))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def load_real(runs, weights):
    parts = [t for t in (run_transitions(r, weights) for r in runs) if t is not None]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fill(buffer, real):
    for i in range(len(real["obs"])):
        buffer.add(real["obs"][i][None], real["next_obs"][i][None], real["action"][i][None],
                   real["reward"][i:i + 1], real["done"][i:i + 1], [{}])


class MixedReplayBuffer(ReplayBuffer):
    """Sample `real_fraction` of each batch from fixed hardware transitions."""

    real = None
    real_fraction = 0.5

    def sample(self, batch_size, env=None):
        n_real = int(round(batch_size * self.real_fraction)) if self.real is not None else 0
        sim = super().sample(batch_size - n_real, env)
        if n_real == 0:
            return sim
        hw = self.real.sample(n_real, env)
        return ReplayBufferSamples(*(None if a is None else torch.cat([a, b]) for a, b in zip(sim, hw)))


class RandomVelocityNoise(ObservationNoise):
    """Per-episode velocity-reading noise.

    The rig's pendulum-velocity reading is noisier at speed than the simulator's
    firmware model (0.85 vs 0.50 rad/s rms while spinning). About 0.8 rad/s of
    extra noise reproduces both hardware outcomes in simulation: LQR passes
    about one in four and SAC settles into a spin.
    """

    def __init__(self, env, low, high, angle_std=0.003, theta_offset=0.0):
        super().__init__(env, angle_std, low)
        self.low, self.high = low, high
        self.theta_offset_max, self.theta_offset = theta_offset, 0.0
        self.level_rng = np.random.default_rng()

    def noisy_frame(self, obs):
        # A small pendulum zero error (calibration or centre-of-mass offset): the
        # policy can only hold upright by also keeping the arm centred.
        frame = super().noisy_frame(obs)
        theta = math.atan2(frame[1], frame[2]) + self.theta_offset
        frame[1:3] = [math.sin(theta), math.cos(theta)]
        return frame

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.level_rng = np.random.default_rng(seed + 53)
        self.velocity_std = float(self.level_rng.uniform(self.low, self.high))
        self.theta_offset = float(self.level_rng.uniform(-self.theta_offset_max, self.theta_offset_max))
        return super().reset(seed=seed, options=options)


class CommandNoise(ObservationNoise):
    """Pendulum reading jitter that grows with the commands just sent.

    Measured on the rig (runs/obs_model): with the 8 ms firmware window and
    device timestamps modelled, the pendulum angle deviates ~0.6 deg rms from
    smooth motion (simulator: 0.045 deg) and the velocity ~0.6 rad/s rms, with no
    extra lag. Both grow with |previous action|, command jumps and pendulum
    speed, are heavy-tailed, and are correlated (0.44). Coefficients are mean
    absolute residuals fitted on runs before V1; `scale` randomises them.
    """

    FIT = Path(__file__).resolve().parents[1] / "runs/obs_model/noise_fit.json"

    def __init__(self, env, scale=(1.0, 1.0), df=4.5):
        super().__init__(env, 0.0, 0.0)
        fit = json.loads(self.FIT.read_text())
        self.cv, self.ca = np.array(fit["velocity_mean_abs"]), np.array(fit["angle_mean_abs"])
        self.rho, self.lag1 = fit["corr_vel_angle"], fit["lag1_velocity"]
        self.scale_range, self.df = scale, df
        # mean |t_df| for unit-scale Student t, so coefficients keep their meaning
        z = np.random.default_rng(0).standard_t(df, 400_000)
        self.t_mean_abs = float(np.mean(np.abs(z)))
        self.scale, self.prev_action, self.prev_u = 1.0, 0.0, 0.0
        self.level_rng = np.random.default_rng()

    def noisy_frame(self, obs):
        frame = np.asarray(obs[-6:], dtype=np.float32).copy()
        a = float(frame[5])
        x = np.array([1.0, abs(a), abs(a - self.prev_action), abs(float(frame[4]))])
        self.prev_action = a
        za, zb = self.rng.standard_t(self.df, 2) / self.t_mean_abs
        u = self.rho * za + math.sqrt(1 - self.rho ** 2) * zb
        ev = self.scale * float(x @ self.cv) * (u + self.lag1 * self.prev_u)
        self.prev_u = u
        theta = math.atan2(frame[1], frame[2]) + self.scale * float(x @ self.ca) * za
        frame[1:3] = [math.sin(theta), math.cos(theta)]
        frame[4] += ev
        return frame

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.level_rng = np.random.default_rng(seed + 97)
        self.scale = float(self.level_rng.uniform(*self.scale_range))
        self.prev_action, self.prev_u = 0.0, 0.0
        return super().reset(seed=seed, options=options)


class RandomActuator(gym.Wrapper):
    """Per-episode physical arm lag behind the step count and stepper zero-crossing kick.

    Dithered-LQR identification on the rig puts the pendulum's response about
    one tick later than the step count's (lag centroid 2.06 ticks; the
    simulator matches with 10-18 ms of arm lag), and with 18-22 ms the
    simulator reproduces the rig's outcomes: LQR holds some captures, every
    SAC balance law fails. Reversal captures fit a 0.2-0.3 rad/s kick.
    """

    def __init__(self, env, lag, kick):
        super().__init__(env)
        self.lag_range, self.kick_range = lag, kick
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng(seed + 131)
        plant = self.env.unwrapped
        plant._arm_lag_s = float(self.rng.uniform(*self.lag_range))
        plant._accel_transport.zero_cross_kick_rad_s = float(self.rng.uniform(*self.kick_range))
        plant.seed_measurement_ring()
        return obs, info


class RandomTilt(gym.Wrapper):
    """Per-episode base tilt: gravity leans up to `max_deg` in a random direction.

    On the rig the pendulum's balance point moves with arm angle (about -0.5 deg
    at -70 deg arm, +0.8 deg at +35 deg), as a slightly tilted base would cause;
    the drift it causes walks the arm into the travel limiter.
    """

    def __init__(self, env, max_deg):
        super().__init__(env)
        self.max_tilt = math.radians(max_deg)
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng(seed + 71)
        tilt, heading = self.rng.uniform(0, self.max_tilt), self.rng.uniform(0, 2 * math.pi)
        g = float(np.linalg.norm(self.env.unwrapped.model.opt.gravity))
        self.env.unwrapped.model.opt.gravity[:] = (g * math.sin(tilt) * math.cos(heading),
                                                   g * math.sin(tilt) * math.sin(heading),
                                                   -g * math.cos(tilt))
        return obs, info


def td_error(model, data, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["obs"]), size=min(n, len(data["obs"])), replace=False)
    with torch.no_grad():
        obs = torch.as_tensor(data["obs"][idx]); nxt = torch.as_tensor(data["next_obs"][idx])
        act = torch.as_tensor(data["action"][idx]); rew = torch.as_tensor(data["reward"][idx])[:, None]
        done = torch.as_tensor(data["done"][idx])[:, None]
        next_act, logp = model.actor.action_log_prob(nxt)
        q_next = torch.min(torch.cat(model.critic_target(nxt, next_act), 1), 1, keepdim=True)[0]
        alpha = torch.exp(model.log_ent_coef.detach())
        target = rew + (1 - done) * model.gamma * (q_next - alpha * logp[:, None])
        q = torch.cat(model.critic(obs, act), 1).mean(1, keepdim=True)
        return float(torch.sqrt(torch.mean((q - target) ** 2)))


class Monitor(BaseCallback):
    def __init__(self, train, held, every, out):
        super().__init__()
        self.train_data, self.held, self.every, self.out, self.log = train, held, every, out, []

    def _on_step(self):
        if self.n_calls % self.every == 0:
            row = {"steps": int(self.num_timesteps),
                   "td_train_real": td_error(self.model, self.train_data),
                   "td_heldout_real": td_error(self.model, self.held)}
            self.log.append(row)
            print(json.dumps(row), flush=True)
            self.model.save(self.out / "checkpoints" / f"sac_{self.num_timesteps}_steps")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--real-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--every", type=int, default=20_000)
    parser.add_argument("--velocity-noise", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Per-episode pendulum/arm velocity reading noise, rad/s")
    parser.add_argument("--theta-offset-deg", type=float, default=0.0,
                        help="Per-episode pendulum zero error, uniform in +-this (needs --velocity-noise)")
    parser.add_argument("--command-noise", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Scale range for the fitted command-dependent reading jitter")
    parser.add_argument("--arm-lag", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Per-episode physical arm lag behind the step count, seconds")
    parser.add_argument("--kick", type=float, nargs=2, default=(0.0, 0.0), metavar=("LOW", "HIGH"),
                        help="Per-episode stepper zero-crossing kick, rad/s")
    parser.add_argument("--tilt-deg", type=float, default=0.0,
                        help="Per-episode base tilt, up to this many degrees in a random direction")
    parser.add_argument("--extra-runs", nargs="*", default=[], help="Further run folders to include")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()

    env = stress_env(Path("configs/my_rig.json"), RIG)
    plant = env.unwrapped
    plant.exact_hanging_reset_frac = 0.35
    plant.upright_reset_frac = 0.25
    weights = plant._reward_weights
    if args.arm_lag or any(args.kick):
        env = RandomActuator(env, tuple(args.arm_lag or (0.0, 0.0)), tuple(args.kick))
    env = RandomDelay(env, 0.006, 0.02)
    env = PendulumDisturbance(env, 6.0, randomize=True)
    if args.tilt_deg:
        env = RandomTilt(env, args.tilt_deg)
    if args.velocity_noise:
        env = RandomVelocityNoise(env, *args.velocity_noise,
                                  theta_offset=math.radians(args.theta_offset_deg))
    if args.command_noise:
        env = CommandNoise(env, tuple(args.command_noise))
    env = gym.wrappers.TransformReward(LimitedActions(env), lambda r: r * REWARD_SCALE)

    runs = sorted(set(default_runs() + args.extra_runs))
    held_runs = runs[::5]
    train_runs = [r for r in runs if r not in held_runs]
    train, held = load_real(train_runs, weights), load_real(held_runs, weights)
    print(f"real transitions: train {len(train['obs'])} from {len(train_runs)} runs, "
          f"held-out {len(held['obs'])} from {len(held_runs)} runs", flush=True)

    model = SAC.load(args.checkpoint, env=env, device="cpu", learning_rate=args.learning_rate,
                     replay_buffer_class=MixedReplayBuffer)
    model.set_random_seed(args.seed)
    real_buffer = ReplayBuffer(len(train["obs"]) + 10, env.observation_space, env.action_space,
                               device="cpu", handle_timeout_termination=False)
    fill(real_buffer, train)
    model.replay_buffer.real = real_buffer if args.real_fraction > 0 else None
    model.replay_buffer.real_fraction = args.real_fraction
    model.learning_starts = model.num_timesteps + 5_000

    monitor = Monitor(train, held, args.every, args.output)
    print(json.dumps({"steps": int(model.num_timesteps), "td_train_real": td_error(model, train),
                      "td_heldout_real": td_error(model, held)}), flush=True)
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=monitor)
    model.save(args.output / "policy.zip")
    (args.output / "training.json").write_text(json.dumps({
        "method": "RLPD-style mixed replay: SAC updates on half hardware, half simulator transitions",
        "source": str(args.checkpoint),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "real_fraction": args.real_fraction, "steps": args.steps, "seed": args.seed,
        "velocity_noise": args.velocity_noise, "theta_offset_deg": args.theta_offset_deg,
        "tilt_deg": args.tilt_deg, "command_noise": args.command_noise,
        "arm_lag": args.arm_lag, "kick": list(args.kick),
        "learning_rate": args.learning_rate, "train_runs": train_runs, "held_out_runs": held_runs,
        "real_transitions": {"train": len(train["obs"]), "held_out": len(held["obs"])},
        "monitor": monitor.log, "rig": RIG, "params": "configs/my_rig.json",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
