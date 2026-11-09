"""Simulator batch with a random pendulum disturbance sized to match the real rig.

The plain simulator is far calmer than the hardware: during balance the real
pendulum shows about 16 rad/s^2 (robust estimate) of acceleration that the
model does not explain, against 3 rad/s^2 in simulation. Adding a random
pendulum acceleration of 9 rad/s^2 RMS (correlated over about 20 ms) and a
15 ms command delay reproduces what the rig does with the LQR reference
preset: about 20% passes, a ~4 degree balance wobble, the arm peaking near its
4.7 rad/s limit, and command RMS near 0.4. Use this batch, not the
undisturbed one, to compare balance settings. Simulation only.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/disturbance_sim.py \
        --algorithm mpc --config configs/mpc_my_rig_low_r.json
"""
import argparse
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from pendulum_lab.controllers import load_controller
from pendulum_lab.environment import make_env, true_state
from pendulum_lab.evaluation import summarize

CALIBRATED_ACCEL_RMS = 9.0     # rad/s^2
CALIBRATED_CORRELATION = 0.3   # per 20 ms tick
CALIBRATED_DELAY_S = 0.015


class PendulumDisturbance(gym.Wrapper):
    """Random, slightly correlated angular acceleration applied to the pendulum."""

    def __init__(self, env, accel_rms, correlation):
        super().__init__(env)
        self.accel_rms, self.correlation = accel_rms, correlation
        self.rng, self.value = np.random.default_rng(0), 0.0

    def reset(self, **kwargs):
        self.rng = np.random.default_rng((kwargs.get("seed") or 0) + 7)
        self.value = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        plant = self.env.unwrapped
        self.value = (self.correlation * self.value
                      + math.sqrt(1 - self.correlation**2) * self.rng.normal() * self.accel_rms)
        plant.data.qfrc_applied[plant._pen_qvel_addr] = self.value * plant.params.pendulum_inertia_kg_m2
        return self.env.step(action)


def batch(algorithm, config, *, params, checkpoint, seeds, seed, accel_rms, correlation, delay_s):
    controller = load_controller(algorithm, params, config, checkpoint)
    env = make_env(params)
    env.unwrapped._fixed_transport_delay_s = delay_s
    env = PendulumDisturbance(env, accel_rms, correlation)
    results = []
    for episode in range(seeds):
        obs, _ = env.reset(seed=seed + episode)
        controller.reset()
        rows, reason = [], "duration"
        for tick in range(750):
            action = float(controller.act(obs))
            obs, _, terminated, _, info = env.step(np.array([action], dtype=np.float32))
            rows.append([(tick + 1) * 0.02, *true_state(env), action])
            if terminated:
                reason = "arm_guard" if info.get("lab_guard") else "terminated"
                break
        results.append(summarize(rows, reason, 15))
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("configs/my_rig.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("docs/results/sac_focused_policy.zip"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--accel-rms", type=float, default=CALIBRATED_ACCEL_RMS)
    parser.add_argument("--correlation", type=float, default=CALIBRATED_CORRELATION)
    parser.add_argument("--delay", type=float, default=CALIBRATED_DELAY_S)
    args = parser.parse_args()
    results = batch(args.algorithm, args.config, params=args.params, checkpoint=args.checkpoint,
                    seeds=args.seeds, seed=args.seed, accel_rms=args.accel_rms,
                    correlation=args.correlation, delay_s=args.delay)
    holds = [r["longest_balance_s"] for r in results]
    print(f"{args.config.name}: passed {sum(r['success'] for r in results)}/{len(results)}, "
          f"median longest balance {np.median(holds):.2f} s, "
          f"guard stops {sum(r['stop_reason'] != 'duration' for r in results)} "
          f"(disturbance {args.accel_rms} rad/s^2, delay {args.delay * 1e3:.0f} ms)")


if __name__ == "__main__":
    main()
