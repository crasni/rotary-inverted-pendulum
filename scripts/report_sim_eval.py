"""Multi-seed simulator evaluation for Report.md (held-out hanging starts, near-upright starts).

Runs the official `pendulum-lab sim` for each method with the same preset and checkpoint as
its hardware pass, then computes per-episode metrics from the episode NPZ traces with the same
definitions as docs/results/report_sim_my_rig.json and report_rl_passes.json.

    python scripts/report_sim_eval.py --check-seed0        # reproduce the committed seed-0 numbers
    python scripts/report_sim_eval.py --mode upright       # 20 near-upright starts, seeds 93000-93019
    python scripts/report_sim_eval.py --mode stress        # held-out stress batch, seeds 92000-92009

The standard hanging start is exact (environment.py sets exact_hanging_reset_frac=1.0), so
every seed gives the same hanging-start episode. The held-out test therefore uses the
randomized stress cases of scripts/transfer_stress.py (varied starts, 20/30 ms delay, sensor
noise, combined) on seeds never used during development, with the final presets/checkpoints.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HZ = 50
SAC_SWING = "docs/results/sac_focused_policy.zip"
METHODS = {
    "pid": ("configs/pid_my_rig_lqr_match.json", SAC_SWING),
    "lqr": ("configs/lqr_my_rig_coupling_delay.json", SAC_SWING),
    "ilqr": ("configs/ilqr_my_rig_lqr_match.json", SAC_SWING),
    "mpc": ("configs/mpc_my_rig_transfer.json", SAC_SWING),
    "ppo": ("configs/sac_my_rig.json", "docs/results/rig_rl_disturbance/ppo_ecG/policy.zip"),
    "sac": ("configs/sac_my_rig.json", "docs/results/rig_rl_disturbance/sac_rnA_reward_only/policy.zip"),
    "dqn": ("configs/sac_my_rig.json", "docs/results/rig_rl_disturbance/dqn_rn05/policy.zip"),
}
CLASSICAL = ["pid", "lqr", "ilqr", "mpc"]
MODES = {
    "upright": dict(start="upright", seed=93000, episodes=20, methods=CLASSICAL),
    "seed0": dict(start="hanging", seed=0, episodes=1, methods=list(METHODS)),
}


def episode_metrics(npz_path, summary):
    x = np.load(npz_path)["samples"]
    theta, theta_dot, arm, action = x[:, 2], x[:, 4], x[:, 1], x[:, 5]
    near = np.flatnonzero(np.abs(theta) <= math.radians(15))
    last = x[-10 * HZ:]
    return {
        "success": summary["success"],
        "stop_reason": summary["stop_reason"],
        "longest_balance_s": round(summary["longest_balance_s"], 2),
        "first_within_15deg_s": round(float(near[0]) / HZ, 2) if len(near) else None,
        "max_abs_theta_deg": round(float(np.degrees(np.abs(theta)).max()), 2),
        "final10_max_abs_theta_deg": round(float(np.degrees(np.abs(last[:, 2])).max()), 2),
        "final10_max_abs_theta_dot_rad_s": round(float(np.abs(last[:, 4]).max()), 2),
        "max_arm_deg": round(float(np.degrees(np.abs(arm)).max()), 1),
        "final10_max_arm_deg": round(float(np.degrees(np.abs(last[:, 1])).max()), 1),
        "command_rms": round(float(np.sqrt(np.mean(action ** 2))), 3),
        "final10_command_rms": round(float(np.sqrt(np.mean(last[:, 5] ** 2))), 3),
    }


def run(method, start, seed, episodes, out_root):
    config, checkpoint = METHODS[method]
    out = out_root / method
    cmd = ["pendulum-lab", "sim", "--algorithm", method, "--params", "configs/my_rig.json",
           "--config", config, "--checkpoint", checkpoint, "--start", start,
           "--duration", "15", "--seed", str(seed), "--episodes", str(episodes),
           "--output", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    summaries = json.loads((out / "summary.json").read_text())
    eps = [episode_metrics(out / f"episode_{i:03d}.npz", s) for i, s in enumerate(summaries)]
    return " ".join(cmd[:-2]), eps


def aggregate(eps):
    def stat(key, fn):
        vals = [e[key] for e in eps if e[key] is not None]
        return round(float(fn(vals)), 3) if vals else None
    passed = [e for e in eps if e["success"]]
    agg = {"episodes": len(eps), "passes": len(passed),
           "safety_stops": sum(e["stop_reason"] != "duration" for e in eps),
           "median_first_within_15deg_s": stat("first_within_15deg_s", np.median),
           "median_longest_balance_s": stat("longest_balance_s", np.median),
           "max_arm_deg": stat("max_arm_deg", np.max),
           "median_command_rms": stat("command_rms", np.median)}
    if passed:
        agg["passing_median_final10_max_abs_theta_deg"] = round(float(np.median(
            [e["final10_max_abs_theta_deg"] for e in passed])), 2)
        agg["passing_median_final10_command_rms"] = round(float(np.median(
            [e["final10_command_rms"] for e in passed])), 3)
    return agg


def distinct_counts(cases):
    """Exact hanging start with no noise is deterministic: count such a case once."""
    from transfer_stress import CASES
    passed = episodes = 0
    for name, r in cases.items():
        case = CASES[name]
        random = case.get("varied_start") or case.get("angle_noise") or case.get("velocity_noise")
        n = r["episodes"] if random else 1
        passed += r["passed"] if random else int(r["passed"] == r["episodes"])
        episodes += n
    return {"distinct_passed": passed, "distinct_episodes": episodes}


def stress(seed=92000, episodes=10):
    sys.path.insert(0, str(Path(__file__).parent))
    from transfer_stress import STRESS_CASES, run as stress_run
    result = {"note": "Held-out simulator stress batch: cases from scripts/transfer_stress.py, "
                      "hanging starts except varied_starts, 15 s, full-task metric. delay_20ms and "
                      "delay_30ms use the exact noiseless hanging start, so their 10 seeds are one "
                      "repeated episode; distinct_* counts each of them once.",
              "params": "configs/my_rig.json", "seeds": [seed, seed + episodes - 1],
              "cases": STRESS_CASES, "methods": {}}
    for method, (config, checkpoint) in METHODS.items():
        cases = stress_run(method, config, params=Path("configs/my_rig.json"),
                           checkpoint=Path(checkpoint), episodes=episodes, seed=seed)
        result["methods"][method] = {"config": config, "checkpoint": checkpoint, "cases": cases,
                                     **distinct_counts(cases)}
    dest = Path("docs/results/report_sim_stress.json")
    dest.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", dest, file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["upright", "stress"])
    p.add_argument("--check-seed0", action="store_true")
    p.add_argument("--out-dir", default="runs/report_sim_eval")
    a = p.parse_args()
    mode = "seed0" if a.check_seed0 else a.mode
    if mode is None:
        p.error("give --mode or --check-seed0")
    if mode == "stress":
        return stress()
    m = MODES[mode]
    out_root = Path(a.out_dir) / mode
    result = {"mode": mode, "start": m["start"], "seeds": [m["seed"], m["seed"] + m["episodes"] - 1],
              "params": "configs/my_rig.json", "duration_s": 15, "methods": {}}
    for method in m["methods"]:
        cmd, eps = run(method, m["start"], m["seed"], m["episodes"], out_root)
        result["methods"][method] = {"command": cmd, "summary": aggregate(eps), "episodes": eps}
        print(method, json.dumps(result["methods"][method]["summary"]), file=sys.stderr)
    if mode == "seed0":
        print(json.dumps({k: v["episodes"][0] for k, v in result["methods"].items()}, indent=1))
        return
    dest = Path(f"docs/results/report_sim_{mode}.json")
    dest.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", dest, file=sys.stderr)


if __name__ == "__main__":
    main()
