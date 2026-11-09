"""Perturbed-simulator gate: does a policy survive rig-like model error?

Every gate used so far (the nominal simulator) is passed by LQR and by SAC versions
that fail on the rig, so it predicts nothing. This gate runs each policy from hanging
in a family of simulators, each episode drawing:

  (family "fitted", the default) command delay and arm lag 14-22 ms, arm-to-pendulum
  coupling x0.9-1.1, gravity torque x0.97-1.03, pendulum friction x1-4, stepper kick
  0.15-0.35 rad/s, sensor noise on the pendulum angle and speed: Student-t (df 3)
  at 0.8-1.2x the rig's measured jitter (0.6 deg, 0.6 rad/s rms). "wide" is v1.

The loop mirrors the hardware runner: firmware-like readings (SimClient), the
controller's own arm limiter, the runner's success test. LQR is the reference
(it is never trained here); the gate is only useful if it ranks policies the way
the rig does. Output: passes and mean longest hold per policy, nominal and perturbed.
"""
import argparse
import dataclasses
import json
import math
import multiprocessing as mp
from collections import deque
from pathlib import Path

import numpy as np

PARAMS = Path("configs/my_rig.json")
SAC_CONFIG = Path("configs/sac_my_rig.json")
LQR_CONFIG = Path("configs/lqr_my_rig_coupling_delay.json")
LQR_SWING_UP = Path("docs/results/sac_focused_policy.zip")
RIG_JITTER = {"angle_rad": math.radians(0.6), "pen_vel": 0.6}


FAMILIES = {
    # v1: wide; ranked LQR no better than SAC (LQR is hurt by lag/delay beyond the fit).
    "wide": {"delay_s": (0.012, 0.030), "arm_lag_s": (0.010, 0.030), "coupling": (0.85, 1.15),
             "gravity": (0.95, 1.05), "friction": (1.0, 4.0), "kick": (0.1, 0.4), "noise": (0.5, 1.5)},
    # v2: the replay fit's uncertainty (docs/sim-fit-upright.md) and the rig's measured jitter.
    "fitted": {"delay_s": (0.014, 0.022), "arm_lag_s": (0.014, 0.022), "coupling": (0.9, 1.1),
               "gravity": (0.97, 1.03), "friction": (1.0, 4.0), "kick": (0.15, 0.35), "noise": (0.8, 1.2)},
}


def draw_knobs(rng, family="fitted"):
    return {k: float(rng.uniform(*r)) for k, r in FAMILIES[family].items()}


NOMINAL = {"delay_s": 0.018, "arm_lag_s": 0.018, "coupling": 1.0, "gravity": 1.0, "friction": 1.0,
           "kick": 0.25, "noise": 0.0}


def student(rng, rms):
    # Student-t with 3 degrees of freedom has variance 3; scale to the wanted rms.
    return float(rng.standard_t(3) * rms / math.sqrt(3.0))


def run_episode(controller, knobs, seed, seconds=15.0):
    from pendulum_lab.contracts import PROFILE, observation_frame
    from pendulum_lab.hardware import validate_state
    from pendulum_lab.rig_session import SimClient, episode_metrics, summarize
    noise = knobs["noise"]
    client = SimClient(PARAMS, seed=seed, **{k: v for k, v in knobs.items() if k != "noise"})
    rng = np.random.default_rng(seed + 1)
    controller.reset()
    client.engage_motor()
    previous, last_action = None, 0.0
    history, rows, requested = None, [], []
    set_time = getattr(controller, "set_sample_time_us", None)
    for _ in range(round(seconds * PROFILE.hz)):
        s = client.get_state()
        if noise and previous is not None:  # the start check sees the clean hanging state
            s = dataclasses.replace(s, pendulum_pos_rad=s.pendulum_pos_rad + student(rng, noise * RIG_JITTER["angle_rad"]),
                           pendulum_vel_rad_s=s.pendulum_vel_rad_s + student(rng, noise * RIG_JITTER["pen_vel"]))
        x = validate_state(s, previous, startup=previous is None)
        previous = s.time_us
        frame = observation_frame(x, last_action)
        history = deque([frame] * PROFILE.history, maxlen=PROFILE.history) if history is None else history
        if rows:
            history.append(frame)
        if callable(set_time):
            set_time(s.time_us)
        action = float(np.clip(controller.act(np.concatenate(history)), -1.0, 1.0))
        client.set_acceleration(action * PROFILE.max_accel)
        rows.append([len(rows) * PROFILE.dt, *x, action])
        requested.append(action)
        last_action = action
    rows = np.asarray(rows)
    result = summarize(rows, "duration", len(rows) / PROFILE.hz)
    result.update(episode_metrics(rows, np.asarray(requested)))
    result["revolutions"] = float(np.abs(np.diff(np.unwrap(rows[:, 2]))).sum() / (2 * math.pi))
    return result


_CACHE = {}


def controller_for(name, spec):
    if name not in _CACHE:
        from pendulum_lab.controllers import load_controller
        algorithm, config, checkpoint = spec
        _CACHE[name] = load_controller(algorithm, PARAMS, Path(config), Path(checkpoint))
    return _CACHE[name]


def _task(job):
    name, spec, knobs, seed = job
    m = run_episode(controller_for(name, spec), knobs, seed)
    return name, knobs, {"success": bool(m["success"]), "longest_balance_s": float(m["longest_balance_s"]),
                         "revolutions": float(m["revolutions"])}


def policies(names):
    out = {}
    for n in names:
        if n == "lqr":
            out[n] = ("lqr", str(LQR_CONFIG), str(LQR_SWING_UP))
        elif ":" in n:  # "<algorithm>:<checkpoint path>", e.g. ppo:runs/.../policy.zip
            algorithm, path = n.split(":", 1)
            out[n] = (algorithm, str(SAC_CONFIG), path)
        else:
            path = Path(n) if n.endswith(".zip") else Path(f"runs/curriculum/versions/{n}/policy.zip")
            out[n] = ("sac", str(SAC_CONFIG), str(path))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policies", nargs="+", help="'lqr', a curriculum version name, or a .zip path")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="fitted")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    specs = policies(args.policies)
    rng = np.random.default_rng(args.seed)
    family = [draw_knobs(rng, args.family) for _ in range(args.episodes)]
    jobs = [(n, s, NOMINAL, args.seed) for n, s in specs.items()]
    jobs += [(n, s, k, args.seed + 10 + i) for n, s in specs.items() for i, k in enumerate(family)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_task, jobs, chunksize=1)
    summary = {}
    for n in specs:
        nom = [r for name, k, r in results if name == n and k == NOMINAL]
        per = [r for name, k, r in results if name == n and k != NOMINAL]
        summary[n] = {"nominal_success": nom[0]["success"], "nominal_hold": nom[0]["longest_balance_s"],
                      "perturbed_passes": sum(r["success"] for r in per), "episodes": len(per),
                      "perturbed_mean_hold": float(np.mean([r["longest_balance_s"] for r in per])),
                      "perturbed_mean_revolutions": float(np.mean([r["revolutions"] for r in per]))}
        s = summary[n]
        print(f"{n:12s} nominal {'PASS' if s['nominal_success'] else 'fail'} {s['nominal_hold']:5.1f}s | "
              f"perturbed {s['perturbed_passes']:2d}/{s['episodes']} passes, mean hold {s['perturbed_mean_hold']:5.2f}s, "
              f"revolutions {s['perturbed_mean_revolutions']:5.1f}",
              flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "family": family,
                                           "results": [[n, k, r] for n, k, r in results]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
