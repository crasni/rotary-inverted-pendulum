"""Paired simulator stress batch for the classical transfer presets.

Runs the same robustness cases and seeds for each algorithm/preset pair and
writes one JSON summary. Simulation only; never connects to hardware.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/transfer_stress.py \
        --output docs/results/transfer_stress_20261107.json
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from pendulum_lab.contracts import PROFILE
from pendulum_lab.controllers import load_controller
from pendulum_lab.environment import true_state
from pendulum_lab.evaluation import summarize
from pendulum_lab.robustness import CASES, stress_env, varied_parameters

PRESETS = {
    "lqr": "configs/lqr_my_rig_coupling_delay.json",
    "mpc": "configs/mpc_my_rig_transfer.json",
    "ilqr": "configs/ilqr_my_rig_transfer.json",
    "pid": "configs/pid_my_rig_transfer.json",
}
STRESS_CASES = ["varied_starts", "delay_20ms", "delay_30ms", "noise_mild", "combined"]


def run(algorithm, config, *, params, checkpoint, episodes, seed, cases=STRESS_CASES):
    controller = load_controller(algorithm, params, Path(config), checkpoint)
    results = {}
    for case in cases:
        with tempfile.TemporaryDirectory() as scratch:
            env = stress_env(varied_parameters(params, CASES[case], Path(scratch) / "params.json"),
                             CASES[case])
            records = []
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
                records.append(summarize(rows, reason, 15))
            env.close()
        results[case] = {"passed": sum(r["success"] for r in records), "episodes": episodes,
                         "safety_stops": sum(r["stop_reason"] != "duration" for r in records),
                         "max_arm_deg": max(r["max_arm_deg"] for r in records)}
        print(algorithm, case, results[case], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("configs/my_rig.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("docs/results/sac_focused_policy.zip"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=40000)
    args = parser.parse_args()
    report = {"params": str(args.params), "checkpoint": str(args.checkpoint),
              "episodes_per_case": args.episodes, "seed": args.seed,
              "note": "Simulation only. The LQR reference also passes here but passed 3 of 16 hardware attempts.",
              "presets": PRESETS, "results": {}}
    for algorithm, config in PRESETS.items():
        report["results"][algorithm] = run(algorithm, config, params=args.params,
                                           checkpoint=args.checkpoint,
                                           episodes=args.episodes, seed=args.seed)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
