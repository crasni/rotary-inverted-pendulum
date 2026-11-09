"""Start the rig simulator from states measured on the real rig.

`bank` saves the four-frame observations near upright from hardware runs, for
training resets. `test` puts the simulator into each such state and lets a
policy run for two seconds, which shows whether it recovers from the states the
real rig produced.
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from pendulum_lab.contracts import PROFILE, observation_frame, wrap
from pendulum_lab.controllers import load_controller
from pendulum_lab.robustness import stress_env

RIG = {"delay_s": 0.018, "speed_limit": 4.4}
PARAMS = Path("configs/my_rig.json")
CONFIG = Path("configs/sac_my_rig.json")
THETA_MAX = 0.8  # rad from upright
PEN_VEL_MAX = 15.0


def load_run(run_dir):
    samples = np.load(Path(run_dir) / "episode_000.npz")["samples"]
    return samples[:, 1:5], samples[:, 5]  # state [arm, theta, arm_vel, pen_vel], sent action


def history(states, actions, k):
    """The observation the policy saw at sample k (frame j carries the action sent at j-1)."""
    return np.concatenate([observation_frame(states[j], actions[j - 1]) for j in range(k - 3, k + 1)])


def near_upright(states, *, theta_max=THETA_MAX, last=None):
    stop = len(states) - (last or 0)
    return [k for k in range(PROFILE.history, stop)
            if abs(states[k, 1]) < theta_max and abs(states[k, 3]) < PEN_VEL_MAX]


def make_bank(runs, output):
    obs = [history(s, a, k) for s, a in map(load_run, runs) for k in near_upright(s)]
    np.savez(output, obs=np.asarray(obs, dtype=np.float32))
    return len(obs)


def set_state(env, states, actions, k):
    plant = env.unwrapped
    arm, theta, arm_vel, pen_vel = states[k]
    plant.data.qpos[plant._motor_qpos_addr] = arm
    plant.data.qpos[plant._pen_qpos_addr] = theta + math.pi
    plant.data.qvel[plant._motor_qvel_addr] = arm_vel
    plant.data.qvel[plant._pen_qvel_addr] = pen_vel
    plant.set_actuator_state(arm, planner_velocity_rad_s=arm_vel)
    mujoco.mj_forward(plant.model, plant.data)
    plant.seed_measurement_ring()
    plant._prev_action = float(actions[k - 1])
    plant._obs_history.clear()
    for j in range(k - 3, k):
        plant._obs_history.append(observation_frame(states[j], actions[j - 1]))
    return plant._obs()


def test(algorithm, checkpoint, runs, *, seconds=2.0, stride=3, config=CONFIG):
    controller = load_controller(algorithm, PARAMS, config, checkpoint)
    env = stress_env(PARAMS, RIG)
    env.reset(seed=0)
    plant = env.unwrapped
    steps = round(seconds * PROFILE.hz)
    held, saturated, real_held, speed = [], [], [], []
    for states, actions in map(load_run, runs):
        for k in near_upright(states, theta_max=0.35, last=steps)[::stride]:
            obs = set_state(env, states, actions, k)
            speed.append(abs(states[k, 3]))
            controller.reset()
            ok, sat = True, []
            for _ in range(steps):
                action = controller.act(obs)
                obs, _, terminated, _, _ = env.step(np.array([action], dtype=np.float32))
                theta = wrap(float(plant.data.qpos[plant._pen_qpos_addr]) - math.pi)
                sat.append(abs(action) > 0.95)
                if terminated or abs(theta) > math.pi / 2:
                    ok = False
                    break
            held.append(ok)
            saturated.append(np.mean(sat))
            real_held.append(bool(np.all(np.abs(states[k:k + steps, 1]) < math.pi / 2)))
    env.close()
    held, speed = np.array(held), np.array(speed)
    by_speed = {name: float(held[mask].mean()) if mask.any() else None for name, mask in
                (("slow_lt2", speed < 2), ("mid_2to4", (speed >= 2) & (speed < 4)), ("fast_ge4", speed >= 4))}
    return {"states": len(held), "sim_held": float(held.mean()), "held_by_speed": by_speed,
            "real_held": float(np.mean(real_held)), "full_strength": float(np.mean(saturated))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bank")
    b.add_argument("runs", nargs="+", type=Path)
    b.add_argument("--output", type=Path, required=True)
    t = sub.add_parser("test")
    t.add_argument("runs", nargs="+", type=Path)
    t.add_argument("--checkpoint", type=Path, action="append", required=True)
    t.add_argument("--algorithm", default="sac")
    args = parser.parse_args()
    if args.command == "bank":
        print(f"{make_bank(args.runs, args.output)} states -> {args.output}")
    else:
        for checkpoint in args.checkpoint:
            print(checkpoint, json.dumps(test(args.algorithm, checkpoint, args.runs)), flush=True)


if __name__ == "__main__":
    main()
