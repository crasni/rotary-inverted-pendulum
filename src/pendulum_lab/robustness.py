"""Reproducible simulator stress tests; never connects to hardware."""
import argparse
from collections import deque
import json
import hashlib
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from .contracts import PARAMS, PROFILE, checked_action, manifest
from .controllers import load_controller
from .environment import make_env, true_state
from .evaluation import COLUMNS, summarize


CASES = {
    "nominal_exact": {},
    "varied_starts": {"varied_start": True},
    "friction_low": {"friction_factor": 0.5},
    "friction_high": {"friction_factor": 1.5},
    "inertia_low": {"inertia_factor": 0.8},
    "inertia_high": {"inertia_factor": 1.2},
    "delay_20ms": {"delay_s": 0.020},
    "delay_30ms": {"delay_s": 0.030},
    "noise_mild": {"angle_noise": 0.003, "velocity_noise": 0.1},
    "noise_high": {"angle_noise": 0.01, "velocity_noise": 0.3},
    "motor_weaker": {"max_accel": 40.0, "speed_limit": 4.0},
    "combined": {"varied_start": True, "friction_factor": 1.5,
                 "inertia_factor": 1.1, "delay_s": 0.020,
                 "angle_noise": 0.003, "velocity_noise": 0.1,
                 "max_accel": 45.0, "speed_limit": 4.5},
}


class ObservationNoise(gym.Wrapper):
    """Additional seeded measurement noise with coherent four-frame history.

    Noise is applied to angles before encoding sin/cos; action history and
    simulator truth used for reward/success metrics are untouched.
    """
    def __init__(self, env, angle_std, velocity_std):
        super().__init__(env)
        self.angle_std = float(angle_std)
        self.velocity_std = float(velocity_std)
        if (not np.isfinite([self.angle_std, self.velocity_std]).all()
                or min(self.angle_std, self.velocity_std) < 0):
            raise ValueError("Noise standard deviations must be finite and non-negative")
        self.history = deque(maxlen=PROFILE.history)
        self.rng = np.random.default_rng()

    def noisy_frame(self, obs):
        frame = np.asarray(obs[-6:], dtype=np.float32).copy()
        frame[0] += self.rng.normal(0, self.angle_std)
        theta = math.atan2(frame[1], frame[2]) + self.rng.normal(0, self.angle_std)
        frame[1:3] = [math.sin(theta), math.cos(theta)]
        frame[3:5] += self.rng.normal(0, self.velocity_std, size=2)
        return frame

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng(seed + 1000000)
        frame = self.noisy_frame(obs)
        self.history.clear()
        self.history.extend([frame.copy() for _ in range(PROFILE.history)])
        return np.concatenate(self.history), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(self.noisy_frame(obs))
        return np.concatenate(self.history), reward, terminated, truncated, info


def varied_parameters(params, case, path):
    doc = json.loads(Path(params).read_text())
    derived = doc["pendulum"]["derived"]
    for key in ("viscous_friction_N_m_s", "coulomb_friction_N_m"):
        if key in derived:
            derived[key] *= case.get("friction_factor", 1.0)
    if "inertia_factor" in case:
        # Resolve CAD fallback through the canonical loader before scaling J.
        from .plant.pendulum_env import PendulumParams
        derived["inertia_measured_kg_m2"] = (
            PendulumParams.load(params).pendulum_inertia_kg_m2 * case["inertia_factor"])
    doc.setdefault("run_meta", {})["robustness_variation"] = case
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")
    return Path(path)


def stress_env(params, case):
    env = make_env(params, start="hanging")
    plant = env.unwrapped
    if case.get("varied_start"):
        plant.exact_hanging_reset_frac = 0.0
    # The canonical reset applies these settings to its planner on every reset.
    plant._fixed_transport_delay_s = case.get("delay_s", 0.010)
    plant._fixed_motor_max_accel_rad_s2 = case.get("max_accel", 50.0)
    plant.motor_planner_speed_limit_rad_s = case.get("speed_limit", 5.0)
    if case.get("angle_noise", 0) or case.get("velocity_noise", 0):
        env = ObservationNoise(env, case.get("angle_noise", 0), case.get("velocity_noise", 0))
    return env


def run_suite(policies, *, params, output, episodes=10, seed=30000, cases=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    selected = CASES if cases is None else (cases if isinstance(cases, dict)
                                           else {name: CASES[name] for name in cases})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"contract": manifest(params), "episodes_requested": episodes, "seed": seed,
              "limitations": "Synthetic uncertainty stress tests, not calibrated hardware bounds. One training seed per policy. Exact noiseless hanging resets are deterministic; repeated copies are evaluated once.",
              "policies": {k: str(v) for k, v in policies.items()},
              "policy_sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in policies.items()},
              "results": []}
    for name, checkpoint in policies.items():
        controller = load_controller(name, params, checkpoint=Path(checkpoint))
        for label, case in selected.items():
            directory = output / name / label
            directory.mkdir(parents=True)
            case_params = varied_parameters(params, case, directory / "params.json")
            env = stress_env(case_params, case)
            records, timings = [], []
            # Seed changes do not vary exact stationary starts without sensor noise.
            count = episodes if case.get("varied_start") or case.get("angle_noise") or case.get("velocity_noise") else 1
            try:
                for episode in range(count):
                    obs, _ = env.reset(seed=seed + episode)
                    initial = true_state(env).tolist()
                    controller.reset()
                    rows, reason = [], "duration"
                    for tick in range(15 * PROFILE.hz):
                        before = time.perf_counter()
                        action = checked_action(controller.act(obs))
                        timings.append((time.perf_counter() - before) * 1000)
                        obs, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
                        rows.append([(tick + 1) * PROFILE.dt, *true_state(env), action])
                        if terminated:
                            reason = "arm_guard" if info.get("lab_guard") else "terminated"
                            break
                        if truncated:
                            break
                    record = summarize(rows, reason, 15)
                    record.update(seed=seed + episode, initial_state=initial)
                    records.append(record)
                    np.savez_compressed(directory / f"episode_{episode:03d}.npz",
                                        samples=rows, columns=COLUMNS)
            finally:
                env.close()
            (directory / "episodes.json").write_text(json.dumps(records, indent=2) + "\n")
            summary = {"algorithm": name, "case": label, "settings": case,
                       "episodes": count, "successes": sum(r["success"] for r in records),
                       "safety_stops": sum(r["stop_reason"] != "duration" for r in records),
                       "mean_longest_balance_s": float(np.mean([r["longest_balance_s"] for r in records])),
                       "max_arm_deg": max(r["max_arm_deg"] for r in records),
                       "action_latency_ms": {"p50": float(np.percentile(timings, 50)),
                                             "p99": float(np.percentile(timings, 99)),
                                             "max": max(timings)},
                       "action_deadline_misses": int(np.sum(np.asarray(timings) > 1000 * PROFILE.dt))}
            report["results"].append(summary)
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, help="algorithm=checkpoint")
    parser.add_argument("--params", type=Path, default=PARAMS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--case", action="append", choices=list(CASES))
    args = parser.parse_args()
    policies = {}
    for item in args.policy:
        name, checkpoint = item.split("=", 1)
        if name not in ("ppo", "sac", "dqn") or name in policies:
            parser.error("Specify each learned algorithm at most once")
        policies[name] = Path(checkpoint)
    run_suite(policies, params=args.params, output=args.output,
              episodes=args.episodes, seed=args.seed, cases=args.case)


if __name__ == "__main__":
    main()
