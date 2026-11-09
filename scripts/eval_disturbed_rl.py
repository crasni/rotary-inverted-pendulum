"""Paired RL evaluation with the teammate's hardware-inspired disturbance.

Use validation seeds to select checkpoints, then a separate seed block for tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from pendulum_lab.contracts import PROFILE
from pendulum_lab.controllers import load_controller
from pendulum_lab.disturbance import PendulumDisturbance
from pendulum_lab.environment import true_state
from pendulum_lab.evaluation import summarize
from pendulum_lab.robustness import CASES, stress_env, varied_parameters
from train_rig_rl import RIG

SCENARIOS = {
    "rig_varied": ({**RIG, "varied_start": True}, 0),
    "disturbance_9": ({**RIG, "varied_start": True}, 9),
    "disturbance_12": ({**RIG, "varied_start": True}, 12),
    "combined": (CASES["combined"], 0),
}


def evaluate(algorithm, checkpoint, *, episodes, seed):
    params = Path("configs/my_rig.json")
    config = Path("configs/sac_my_rig.json")
    controller = load_controller(algorithm, params, config, checkpoint)
    results = {}
    for name, (case, rms) in SCENARIOS.items():
        with tempfile.TemporaryDirectory() as scratch:
            env = PendulumDisturbance(stress_env(varied_parameters(
                params, case, Path(scratch) / "params.json"), case), rms)
            records = []
            try:
                for episode in range(episodes):
                    obs, _ = env.reset(seed=seed + episode)
                    controller.reset()
                    rows, reason = [], "duration"
                    for tick in range(15 * PROFILE.hz):
                        action = float(controller.act(obs))
                        obs, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
                        rows.append([(tick + 1) * PROFILE.dt, *true_state(env), action])
                        if terminated:
                            reason = "arm_guard" if info.get("lab_guard") else "terminated"
                            break
                        if truncated:
                            break
                    record = summarize(rows, reason, 15)
                    records.append({**record, "seed": seed + episode})
            finally:
                env.close()
        results[name] = {
            "passed": sum(r["success"] for r in records), "episodes": episodes,
            "safety_stops": sum(r["stop_reason"] != "duration" for r in records),
            "mean_longest_balance_s": float(np.mean([r["longest_balance_s"] for r in records])),
            "max_arm_deg": max(r["max_arm_deg"] for r in records), "records": records,
        }
        print(algorithm, checkpoint, name, results[name]['passed'], '/', episodes, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, help="label:algorithm=checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=71000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Require positive episodes and nonnegative seed")
    policies = {}
    for item in args.policy:
        label, spec = item.split(":", 1)
        algorithm, path = spec.split("=", 1)
        if label in policies or algorithm not in ("ppo", "dqn", "sac"):
            parser.error("Require unique labels and PPO/DQN/SAC algorithms")
        policies[label] = (algorithm, Path(path))
    report = {"seed": args.seed, "episodes": args.episodes, "scenarios": SCENARIOS,
              "config": "configs/sac_my_rig.json", "params": "configs/my_rig.json",
              "results": {}, "note": "Simulation only; surrogate disturbance is not hardware validation."}
    report["input_sha256"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in (report["config"], report["params"])}
    with args.output.open("x") as stream:
        for label, (algorithm, path) in policies.items():
            report["results"][label] = {
                "algorithm": algorithm, "checkpoint": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cases": evaluate(algorithm, path, episodes=args.episodes, seed=args.seed),
            }
            stream.seek(0)
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.truncate()
            stream.flush()


if __name__ == "__main__":
    main()
