"""Fit the rig simulator's near-upright dynamics to the rig's own logs.

For many short segments of real episodes (start within --max-deg of upright),
put the simulator into the logged state and replay the exact commands the rig
received for --horizon ticks, open loop. The error is the gap between the
simulated and the logged pendulum angle (and arm step count) over the segment.
Knobs fitted (all existing, opt-in simulator options):

  delay_s        acceleration-command transport delay
  arm_lag_s      physical arm lag behind the step count
  coupling       fraction of the arm-to-pendulum inertial coupling that acts
  kick           stepper zero-crossing kick (rad/s)
  friction       pendulum viscous and Coulomb friction scale
  gravity        scale on the pendulum's gravity torque (upright fall rate)
  arm_hz/zeta    bandwidth and damping of the arm's PD servo to its step count
                 (None = the simulator's stiff default): how the physical arm
                 follows abrupt acceleration changes

The loss is a trimmed mean over segments (the worst 5% dropped), so a few
segments where the pendulum was touched or stuck cannot steer the fit.
"""
import argparse
import dataclasses
import glob
import json
import math
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

PARAMS = Path("configs/my_rig.json")
BASE = {"delay_s": 0.018, "arm_lag_s": 0.018, "coupling": 1.0, "kick": 0.25, "friction": 1.0, "gravity": 1.0,
        "arm_hz": None, "arm_zeta": 1.0}
DEFAULT_SOURCES = ["runs/curriculum/[ebs]*/episode_*.npz", "runs/lqr*/episode_*.npz",
                   "docs/results/hardware/lqr/*/episode_*.npz"]


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def segments(patterns, horizon, max_deg, stride, max_pen_vel=4.0, max_arm_deg=60.0):
    """(file, start, states[horizon+1, 4], sent[horizon+1]) near upright."""
    out = []
    for f in sorted({f for p in patterns for f in glob.glob(p)}):
        try:
            x = np.load(f, allow_pickle=True)["samples"]
        except (OSError, KeyError, ValueError):
            continue
        if len(x) < horizon + 10 or not 0.018 < np.median(np.diff(x[:, 0])) < 0.022:
            continue
        theta = np.vectorize(wrap)(x[:, 2])
        for k in range(2, len(x) - horizon - 1, stride):
            if (abs(theta[k]) < math.radians(max_deg) and abs(x[k, 4]) < max_pen_vel
                    and abs(x[k, 1]) < math.radians(max_arm_deg)
                    and np.all(np.abs(np.diff(x[k:k + horizon + 1, 0]) - 0.02) < 0.006)):
                s = x[k - 1:k + horizon + 1].copy()
                s[:, 2] = theta[k - 1:k + horizon + 1]
                out.append((f, k, s[:, 1:5], s[:, 5]))
    return out


def make_env(knobs):
    from pendulum_lab.plant.pendulum_env import RotaryInvertedPendulumEnv
    env = RotaryInvertedPendulumEnv(
        params_path=PARAMS, control_freq_hz=50, action_mode="accel", action_smooth_window=1,
        max_accel_rad_s2=50.0, motor_max_accel_rad_s2=50.0, transport_delay_s=knobs["delay_s"],
        motor_planner_speed_limit_rad_s=4.4, firmware_obs_model=True, episode_length_s=3600.0,
        terminate_on_spin_loop=False, terminate_on_hard_stop=False)
    env.reset(seed=0)
    env._arm_lag_s = knobs["arm_lag_s"]
    env._coupling_gain = knobs["coupling"]
    env._accel_transport.zero_cross_kick_rad_s = knobs["kick"]
    env.params = dataclasses.replace(env.params, pendulum_friction=env.params.pendulum_friction * knobs["friction"],
                                     pendulum_coulomb=env.params.pendulum_coulomb * knobs["friction"])
    env.model.dof_damping[env._pen_dof_addr] = env.params.pendulum_friction
    env.model.dof_frictionloss[env._pen_dof_addr] = env.params.pendulum_coulomb
    set_arm_servo(env, knobs.get("arm_hz"), knobs.get("arm_zeta", 1.0))
    if knobs["gravity"] != 1.0:
        # Scale the pendulum's own gravity torque: g * m * l * sin enters through
        # the body's mass and centre of mass; scaling the gravity vector would
        # also act on the arm, which is position controlled, so it is harmless.
        env.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81 * knobs["gravity"]])
    return env


def set_arm_servo(env, hz, zeta):
    """Re-tune the arm's position actuator to natural frequency `hz` and damping
    ratio `zeta` against the same effective inertia (default: critical, kp 1000)."""
    if hz is None:
        return
    kp0, kv0 = env.model.actuator_gainprm[0, 0], -env.model.actuator_biasprm[0, 2]
    inertia = kv0 ** 2 / (4.0 * kp0)  # the default kv is critical damping for this inertia
    kp = inertia * (2 * math.pi * hz) ** 2
    kv = 2 * zeta * math.sqrt(kp * inertia)
    env.model.actuator_gainprm[0, 0] = kp
    env.model.actuator_biasprm[0, 1] = -kp
    env.model.actuator_biasprm[0, 2] = -kv


def replay(env, states, sent, d_theta=0.0, d_pen_vel=0.0):
    """Simulated (theta, arm) after each replayed command, from states[1]
    (optionally with the starting angle and pendulum speed shifted)."""
    k = 1
    arm, theta, arm_vel, pen_vel = states[k]
    theta, pen_vel = theta + d_theta, pen_vel + d_pen_vel
    # The logged arm is the step count; the physical arm trails it by the arm
    # lag, so it starts at arm - arm_vel * lag and the lag buffer (seeded forward
    # from the physical arm) ends exactly at the step count.
    env.data.qpos[env._motor_qpos_addr] = arm - arm_vel * env._arm_lag_s
    env.data.qpos[env._pen_qpos_addr] = theta + math.pi
    env.data.qvel[env._motor_qvel_addr] = arm_vel
    env.data.qvel[env._pen_qvel_addr] = pen_vel
    env.set_actuator_state(arm, planner_velocity_rad_s=arm_vel)
    env._accel_transport.active_accel_rad_s2 = float(sent[k - 1]) * 50.0
    mujoco.mj_forward(env.model, env.data)
    env.seed_measurement_ring()
    env._prev_action = float(sent[k - 1])
    th, ar = [], []
    for j in range(k, len(states) - 1):
        env.step(np.array([sent[j]], dtype=np.float32))
        th.append(wrap(float(env.data.qpos[env._pen_qpos_addr]) - math.pi))
        ar.append(env._accel_transport.position_rad)
    return np.array(th), np.array(ar)


# Measurement noise on the starting state (rig: angle ~0.6 deg, speed ~0.6 rad/s
# rms) is amplified by the upright instability (~e^(12 t)), so each segment's
# starting angle and pendulum speed are re-estimated within these bounds.
START_FIT = {"d_theta": math.radians(1.5), "d_pen_vel": 1.5}


def segment_errors(knobs, segs, fit_start=True):
    env = make_env(knobs)
    out = np.empty((len(segs), 2))
    for i, (_, _, states, sent) in enumerate(segs):
        th, ar = replay(env, states, sent)
        target = states[2:, 1]
        if fit_start:
            # The response is close to linear in small starting shifts: two
            # extra replays give its sensitivity; solve a bounded 2-D least squares.
            jt = (replay(env, states, sent, d_theta=1e-3)[0] - th) / 1e-3
            jv = (replay(env, states, sent, d_pen_vel=0.1)[0] - th) / 0.1
            J = np.stack([jt, jv], 1)
            c, *_ = np.linalg.lstsq(J, np.vectorize(wrap)(target - th), rcond=None)
            c = np.clip(c, [-START_FIT["d_theta"], -START_FIT["d_pen_vel"]], [START_FIT["d_theta"], START_FIT["d_pen_vel"]])
            th = th + J @ c
        out[i, 0] = np.sqrt(np.mean(np.vectorize(wrap)(th - target) ** 2))
        out[i, 1] = np.sqrt(np.mean((ar - states[2:, 0]) ** 2))
    env.close()
    return np.degrees(out)


def trimmed(values, drop=0.05):
    v = np.sort(values)
    return float(v[: max(1, int(len(v) * (1 - drop)))].mean())


_SEGS = None


def _init(segs):
    global _SEGS
    _SEGS = segs


def _loss(knobs):
    e = segment_errors(knobs, _SEGS)
    return trimmed(e[:, 0]), trimmed(e[:, 1])


def evaluate(pool, configs):
    return pool.map(_loss, configs)


def coordinate_search(pool, start, grid, rounds):
    """Coordinate descent over the knob grid; one pool.map per knob per round."""
    best = dict(start)
    best_loss = evaluate(pool, [best])[0][0]
    history = [{"knobs": dict(best), "theta_deg": best_loss}]
    for _ in range(rounds):
        improved = False
        for name, values in grid.items():
            configs = [{**best, name: v} for v in values]
            losses = evaluate(pool, configs)
            i = int(np.argmin([l[0] for l in losses]))
            if losses[i][0] < best_loss - 1e-4:
                best, best_loss, improved = configs[i], losses[i][0], True
            history.append({"knob": name, "values": list(values), "theta_deg": [l[0] for l in losses]})
            print(f"  {name:10s} -> {best[name]:.4g}  (trimmed theta rms {best_loss:.3f} deg)", flush=True)
        if not improved:
            break
    return best, best_loss, history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=10, help="Replayed ticks per segment (10 = 0.2 s)")
    parser.add_argument("--max-deg", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-segments", type=int, default=3000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    segs = segments(DEFAULT_SOURCES, args.horizon, args.max_deg, args.stride)
    rng = np.random.default_rng(0)
    files = sorted({s[0] for s in segs})
    held_out = set(rng.choice(files, size=len(files) // 5, replace=False))
    train = [s for s in segs if s[0] not in held_out]
    test = [s for s in segs if s[0] in held_out]
    if len(train) > args.max_segments:
        train = [train[i] for i in sorted(rng.choice(len(train), args.max_segments, replace=False))]
    print(f"segments: {len(train)} train, {len(test)} held-out (episodes split {len(files) - len(held_out)}/{len(held_out)})",
          flush=True)
    grid = {"delay_s": [0.01, 0.014, 0.018, 0.022, 0.026, 0.034, 0.042],
            "arm_lag_s": [0.0, 0.006, 0.012, 0.016, 0.02, 0.026, 0.034],
            "coupling": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "kick": [0.0, 0.15, 0.25, 0.4],
            "friction": [1.0, 2.0, 4.0, 8.0, 16.0],
            "gravity": [0.85, 0.88, 0.91, 0.94, 0.97, 1.0, 1.05]}
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(train,)) as pool:
        best, loss, history = coordinate_search(pool, BASE, grid, args.rounds)
    base_test, best_test = segment_errors(BASE, test), segment_errors(best, test)
    result = {"base": BASE, "fitted": best, "train_theta_deg": loss,
              "held_out": {"segments": len(test),
                           "base_theta_deg": trimmed(base_test[:, 0]), "fitted_theta_deg": trimmed(best_test[:, 0]),
                           "base_arm_deg": trimmed(base_test[:, 1]), "fitted_arm_deg": trimmed(best_test[:, 1])},
              "horizon_ticks": args.horizon, "max_deg": args.max_deg, "history": history}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
