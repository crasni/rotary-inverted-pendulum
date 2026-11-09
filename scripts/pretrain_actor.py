"""Pretrain a SAC actor's balancing on LQR's successful rig balancing.

On the real states where LQR balanced, SAC policies command almost unrelated
actions (sign agreement ~56%), and replaying LQR data through the critic (R50)
did not change that. This step teaches the actor directly before on-rig SAC
learning:

  demonstrations  LQR rig runs with the working preset, ticks in balance mode
                  inside stretches held >= 1 s, without the last 0.3 s before a
                  loss; target = LQR's own command (logged requested action).
  anchor          simulator rollouts of the starting policy away from upright
                  (|theta| > 0.5 rad); target = its own deterministic action, so
                  swing-up is kept.

Only the actor changes. The result is still a SAC policy that acts on its own;
LQR only provides training data. Every fifth LQR run is held out to check that
the imitation generalises.
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

import real_state_starts as R

LQR_CONFIG = "lqr_my_rig_coupling_delay.json"


def lqr_runs():
    runs = []
    for r in sorted(glob.glob("docs/results/hardware/lqr/2026*")):
        args = json.loads(Path(r, "run.json").read_text())["arguments"]
        if args.get("config", "").endswith(LQR_CONFIG) and Path(r, "episode_000.npz").exists():
            runs.append(r)
    return runs


def demonstrations(runs):
    obs, target = [], []
    for r in runs:
        d = np.load(Path(r, "episode_000.npz"), allow_pickle=True)
        s, timing = d["samples"], d["hardware_timing"]
        if len(s) < 60 or len(timing) != len(s):
            continue
        mode, requested = timing[:, 6] == 1, timing[:, 7]
        states, sent = s[:, 1:5], s[:, 5]
        k = 0
        while k < len(s):
            if not mode[k]:
                k += 1
                continue
            end = k
            while end < len(s) and mode[end]:
                end += 1
            if end - k >= 50:
                stop = end - 15 if end < len(s) else end  # drop the fall itself
                for j in range(max(k, 4), stop):
                    if np.isfinite(requested[j]):
                        obs.append(R.history(states, sent, j)); target.append(np.clip(requested[j], -1, 1))
            k = end
    return np.asarray(obs, dtype=np.float32), np.asarray(target, dtype=np.float32)


def anchor_states(checkpoint, episodes=40, seed=7000):
    from pendulum_lab.controllers import load_controller
    from pendulum_lab.robustness import stress_env
    from train_rig_rl import RIG
    c = load_controller("sac", Path("configs/my_rig.json"), Path("configs/sac_my_rig.json"), Path(checkpoint))
    env = stress_env(Path("configs/my_rig.json"), RIG)
    p = env.unwrapped
    p.exact_hanging_reset_frac = 0.5
    obs = []
    for e in range(episodes):
        o, _ = env.reset(seed=seed + e)
        c.reset()
        for _ in range(500):
            theta = math.atan2(o[-5], o[-4])
            if abs(theta) > 0.5:
                obs.append(o.copy())
            o, _, terminated, _, _ = env.step(np.array([c.act(o)], dtype=np.float32))
            if terminated:
                break
    return np.asarray(obs, dtype=np.float32)


def mean_action(actor, obs):
    mean, _, _ = actor.get_action_dist_params(obs)
    return torch.tanh(mean).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    from stable_baselines3 import SAC

    runs = lqr_runs()
    held = runs[::5]
    train = [r for r in runs if r not in held]
    demo_obs, demo_act = demonstrations(train)
    held_obs, held_act = demonstrations(held)
    anchor_obs = anchor_states(args.checkpoint)
    model = SAC.load(args.checkpoint, device="cpu")
    actor = model.policy.actor
    with torch.no_grad():
        anchor_act = mean_action(actor, torch.as_tensor(anchor_obs)).numpy()

    def agreement(o, a):
        with torch.no_grad():
            p = mean_action(actor, torch.as_tensor(o)).numpy()
        return {"corr": float(np.corrcoef(p, a)[0, 1]), "sign": float(np.mean(np.sign(p) == np.sign(a))),
                "mae": float(np.mean(np.abs(p - a)))}

    print(f"demonstrations: train {len(demo_obs)} from {len(train)} runs, held-out {len(held_obs)} "
          f"from {len(held)} runs; anchor states {len(anchor_obs)}", flush=True)
    log = [{"step": 0, "train": agreement(demo_obs, demo_act), "held_out": agreement(held_obs, held_act),
            "anchor": agreement(anchor_obs, anchor_act)}]
    print(json.dumps(log[-1]), flush=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    for step in range(1, args.steps + 1):
        i = rng.integers(0, len(demo_obs), 128)
        j = rng.integers(0, len(anchor_obs), 128)
        loss = torch.mean((mean_action(actor, torch.as_tensor(demo_obs[i])) - torch.as_tensor(demo_act[i])) ** 2) \
            + args.anchor_weight * torch.mean((mean_action(actor, torch.as_tensor(anchor_obs[j]))
                                               - torch.as_tensor(anchor_act[j])) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            log.append({"step": step, "loss": float(loss), "train": agreement(demo_obs, demo_act),
                        "held_out": agreement(held_obs, held_act), "anchor": agreement(anchor_obs, anchor_act)})
            print(json.dumps(log[-1]), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    model.save(args.output / "policy.zip")
    (args.output / "pretraining.json").write_text(json.dumps({
        "source": str(args.checkpoint), "train_runs": train, "held_out_runs": held,
        "demonstrations": len(demo_obs), "anchor_states": len(anchor_obs), "steps": args.steps,
        "learning_rate": args.learning_rate, "anchor_weight": args.anchor_weight, "log": log}, indent=2) + "\n")


if __name__ == "__main__":
    main()
