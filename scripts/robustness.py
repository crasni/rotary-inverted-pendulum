"""Robustness evaluation: run controllers on randomly perturbed simulated rigs.

The controller is always built from --params (the model we believe), while
each simulated rig uses perturbed physics. This measures how a controller
copes with model mismatch, which is what the real pendulum will present.

Example:
    python scripts/robustness.py --algorithms pid lqr ilqr mpc \
        --rigs 30 --workers 8 --output runs/robust_baseline_01
"""
import argparse
import csv
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pendulum_lab.contracts import PARAMS, PROFILE, checked_action
from pendulum_lab.controllers import load_controller
from pendulum_lab.environment import make_env, true_state
from pendulum_lab.evaluation import summarize
from pendulum_lab.plant.pendulum_geometry import PENDULUM_COM_M

# Multiplicative ranges. Mass/inertia/COM uniform; friction and delay log-uniform.
RANGES = {
    "mass": (0.85, 1.15),
    "inertia": (0.85, 1.15),
    "com": (0.90, 1.10),
    "viscous": (1 / 3, 3.0),
    "coulomb": (0.2, 5.0),
    "delay_s": (0.005, 0.030),   # absolute seconds; the lab default is 0.010
}


def sample_rig(base_doc, rng):
    """Return (perturbed JSON document, factors). Rejects non-physical draws."""
    while True:
        f = {
            "mass": rng.uniform(*RANGES["mass"]),
            "inertia": rng.uniform(*RANGES["inertia"]),
            "com": rng.uniform(*RANGES["com"]),
            "viscous": math.exp(rng.uniform(*np.log(RANGES["viscous"]))),
            "coulomb": math.exp(rng.uniform(*np.log(RANGES["coulomb"]))),
            "delay_s": math.exp(rng.uniform(*np.log(RANGES["delay_s"]))),
        }
        doc = json.loads(json.dumps(base_doc))
        meta = doc.setdefault("run_meta", {})
        derived = doc["pendulum"]["derived"]
        mass = float(meta.get("pendulum_total_mass_kg", 0.014)) * f["mass"]
        com = float(meta.get("pendulum_com_m", PENDULUM_COM_M)) * f["com"]
        inertia = float(derived["inertia_measured_kg_m2"]) * f["inertia"]
        if inertia <= 1.05 * mass * com ** 2:   # COM inertia must stay positive
            continue
        meta["pendulum_total_mass_kg"] = mass
        meta["pendulum_com_m"] = com
        derived["inertia_measured_kg_m2"] = inertia
        derived["viscous_friction_N_m_s"] = float(derived["viscous_friction_N_m_s"]) * f["viscous"]
        derived["coulomb_friction_N_m"] = float(derived.get("coulomb_friction_N_m", 0.0)) * f["coulomb"]
        return doc, f


def run_one(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    alg, rig_id, rig_path, factors, a = job
    controller = load_controller(alg, Path(a["params"]),
                                 Path(a["config"]) if a["config"] else None,
                                 Path(a["checkpoint"]) if a["checkpoint"] else None)
    env = make_env(Path(rig_path), duration=a["duration"], start=a["start"])
    # make_env fixes the transport delay at 10 ms; reset() reads this field.
    env.unwrapped._fixed_transport_delay_s = factors["delay_s"]
    rows, times, reason = [], [], "duration"
    try:
        obs, _ = env.reset(seed=a["seed"] + rig_id)
        controller.reset()
        for tick in range(round(a["duration"] * PROFILE.hz)):
            t0 = time.perf_counter()
            action = checked_action(controller.act(obs))
            times.append(time.perf_counter() - t0)
            obs, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
            rows.append([(tick + 1) * PROFILE.dt, *true_state(env), action])
            if terminated:
                reason = "arm_guard" if info.get("lab_guard") else "terminated"
                break
            if truncated:
                break
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    s = summarize(rows, reason, a["duration"])
    ms = np.array(times) * 1000 if times else np.zeros(1)
    return {"algorithm": alg, "rig": rig_id, **{k: round(v, 4) for k, v in factors.items()},
            "success": s["success"], "stop_reason": s["stop_reason"],
            "longest_balance_s": s.get("longest_balance_s", 0.0),
            "max_arm_deg": round(s.get("max_arm_deg", 0.0), 1),
            "act_ms_mean": round(float(ms.mean()), 2), "act_ms_max": round(float(ms.max()), 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--algorithms", nargs="+", default=["pid", "lqr", "ilqr", "mpc"])
    p.add_argument("--params", default=str(PARAMS), help="model the controller believes")
    p.add_argument("--config")
    p.add_argument("--checkpoint")
    p.add_argument("--rigs", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--start", default="hanging", choices=("hanging", "upright", "mixed"))
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    a = vars(p.parse_args())

    out = Path(a["output"])
    out.mkdir(parents=True, exist_ok=False)
    rigs_dir = out / "rigs"
    rigs_dir.mkdir()
    base = json.loads(Path(a["params"]).read_text())
    rng = np.random.default_rng(a["seed"])
    jobs = []
    rig_list = []
    for i in range(a["rigs"]):
        doc, factors = sample_rig(base, rng)
        path = rigs_dir / f"rig_{i:03d}.json"
        path.write_text(json.dumps(doc, indent=2))
        rig_list.append((i, str(path), factors))
    for alg in a["algorithms"]:
        jobs += [(alg, i, path, f, a) for i, path, f in rig_list]
    (out / "run.json").write_text(json.dumps({"args": a, "ranges": RANGES}, indent=2))

    started = time.time()
    with Pool(a["workers"]) as pool:
        results = pool.map(run_one, jobs, chunksize=1)
    with open(out / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    print(f"{a['rigs']} rigs, start={a['start']}, {time.time() - started:.0f} s wall time")
    print(f"{'alg':6} {'success':>9} {'arm_guard':>9} {'med_arm':>8} {'ms_mean':>8} {'ms_max':>7}")
    for alg in a["algorithms"]:
        r = [x for x in results if x["algorithm"] == alg]
        ok = sum(x["success"] for x in r)
        guard = sum(x["stop_reason"] == "arm_guard" for x in r)
        errors = [x["stop_reason"] for x in r if x["stop_reason"] not in ("duration", "arm_guard")]
        summary[alg] = {"success": ok, "n": len(r), "arm_guard": guard, "errors": errors[:3],
                        "median_max_arm_deg": float(np.median([x["max_arm_deg"] for x in r])),
                        "act_ms_mean": float(np.mean([x["act_ms_mean"] for x in r])),
                        "act_ms_max": float(np.max([x["act_ms_max"] for x in r]))}
        s = summary[alg]
        print(f"{alg:6} {ok:>4}/{len(r):<4} {guard:>9} {s['median_max_arm_deg']:>8.1f} "
              f"{s['act_ms_mean']:>8.2f} {s['act_ms_max']:>7.1f}" + (f"  errors: {errors[:1]}" if errors else ""))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
