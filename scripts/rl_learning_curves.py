"""Learning curves and checkpoint sample efficiency for the final PPO, SAC and DQN policies.

    python scripts/rl_learning_curves.py eval   # evaluate saved checkpoints in the standard simulator
    python scripts/rl_learning_curves.py plot   # write docs/results/rl_learning_curves.png (+ .json)

Training reward comes from each fine-tuning stage's own SB3 log (`*.log` tables or
`progress.csv`). Each stage trains in a different environment (noise, delay, energy cap,
reward scale), so rewards are comparable only within a stage, not across stages.
The from-scratch curves (docs/results/learning_curves.csv) are the first training phase of
each method on the starter model, unscaled episode return.
The checkpoint evaluation uses the randomized "combined" stress case of
scripts/transfer_stress.py (varied start, 20 ms delay, sensor noise, friction, inertia and
motor limits) on configs/my_rig.json, 15 s, deterministic policy, seeds 94000-94009,
full-task metric. The nominal hanging start is exact, so it cannot separate checkpoints:
every final-stage checkpoint passes it.
"""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np

V = Path("runs/curriculum/versions")
# (label, kind, path) in lineage order; only stages whose training logs survive.
STAGES = {
    "ppo": [("ppo_ecB", "log", V / "ppo_ecB.log"), ("ppo_ecE", "log", V / "ppo_ecE.log"),
            ("ppo_ecG", "log", V / "ppo_ecG.log")],
    "sac": [("V1", "log", Path("runs/sac_finetune/V1.log")),
            ("v2m2", "csv", V / "v2m2/log/progress.csv"), ("v3b", "csv", V / "v3b/log/progress.csv"),
            ("v6b", "csv", V / "v6b/log/progress.csv"), ("rnA", "csv", V / "rnA/log/progress.csv")],
    "dqn": [("dqn_rn05", "log", V / "dqn_rn05.log")],
}
CHECKPOINTS = {"ppo": V / "ppo_ecG/checkpoints", "dqn": V / "dqn_rn05/checkpoints"}
FINAL = {"ppo": "docs/results/rig_rl_disturbance/ppo_ecG/policy.zip",
         "dqn": "docs/results/rig_rl_disturbance/dqn_rn05/policy.zip"}
SEED, EPISODES = 94000, 10
EVAL_JSON = Path("docs/results/rl_checkpoint_eval.json")


def read_log(path):
    steps = rew = None
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\|\s+(\S+)\s+\|\s+(\S+)", line)
        if m and m.group(1) == "ep_rew_mean":
            rew = float(m.group(2))
        elif m and m.group(1) == "total_timesteps":
            steps = int(m.group(2))
        elif line.startswith("---") and steps is not None and rew is not None:
            out.append((steps, rew))
            steps = rew = None
    return out


def read_csv(path):
    out = []
    for row in csv.DictReader(path.open()):
        if row.get("rollout/ep_rew_mean"):
            out.append((int(float(row["time/total_timesteps"])), float(row["rollout/ep_rew_mean"])))
    return out


def evaluate():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from transfer_stress import run as stress_run
    result = {"case": "combined", "seeds": [SEED, SEED + EPISODES - 1], "duration_s": 15,
              "params": "configs/my_rig.json", "config": "configs/sac_my_rig.json", "methods": {}}
    for alg, folder in CHECKPOINTS.items():
        ckpts = sorted(folder.glob(f"{alg}_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
        final_steps = json.loads((folder.parent / "training.json").read_text())["final_steps"]
        points = []
        for ckpt, steps in [*[(c, int(c.stem.split("_")[1])) for c in ckpts], (Path(FINAL[alg]), final_steps)]:
            r = stress_run(alg, "configs/sac_my_rig.json", params=Path("configs/my_rig.json"),
                           checkpoint=ckpt, episodes=EPISODES, seed=SEED, cases=["combined"])["combined"]
            points.append({"checkpoint": ckpt.name, "steps": steps, "passes": r["passed"],
                           "episodes": r["episodes"], "safety_stops": r["safety_stops"]})
        result["methods"][alg] = points
    EVAL_JSON.write_text(json.dumps(result, indent=1) + "\n")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = json.loads(EVAL_JSON.read_text())["methods"] if EVAL_JSON.exists() else {}
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 3, 2]})
    curves = {}
    scratch = {}
    for row in csv.DictReader(open("docs/results/learning_curves.csv")):
        scratch.setdefault(row["algorithm"], []).append(
            (int(row["total_steps_at_episode_end"]), float(row["unscaled_episode_return"])))
    for col, alg in enumerate(["ppo", "sac", "dqn"]):
        x, y = np.array(sorted(scratch[alg])).T
        k = max(1, len(y) // 60)
        ax = axes[0, col]
        ax.plot(x / 1e6, y, ".", ms=2, color="0.75")
        ax.plot(x[k - 1:] / 1e6, np.convolve(y, np.ones(k) / k, mode="valid"), lw=1.5)
        ax.set_title(f"{alg.upper()}: training from scratch (starter model)")
        ax.set_xlabel("environment steps (M)")
        ax.set_ylabel("episode return (unscaled)")
        ax = axes[1, col]
        curves[alg] = {}
        for label, kind, path in STAGES[alg]:
            pts = read_log(path) if kind == "log" else read_csv(path)
            if not pts:
                continue
            x, y = np.array(pts).T
            curves[alg][label] = {"source": str(path), "steps": x.astype(int).tolist(), "ep_rew_mean": y.tolist()}
            ax.plot(x / 1e6, y, lw=1.2, label=label)
            ax.axvline(x[0] / 1e6, color="0.8", lw=0.8, zorder=0)
        ax.set_title(f"{alg.upper()}: training reward per fine-tuning stage")
        ax.set_xlabel("cumulative environment steps (M)")
        ax.set_ylabel("ep_rew_mean (stage's own env and scale)")
        ax.legend(fontsize=8)
        ax = axes[2, col]
        if alg in ev:
            pts = ev[alg]
            ax.plot([p["steps"] / 1e6 for p in pts], [p["passes"] / p["episodes"] for p in pts], "o-", ms=3)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(f"pass rate, combined stress ({pts[0]['episodes']} seeds)")
            ax.set_xlabel("cumulative environment steps (M)")
            ax.set_title(f"{alg.upper()}: final-stage checkpoints under stress")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No intermediate checkpoints were saved\nfor SAC rnA", ha="center", va="center")
    fig.tight_layout()
    fig.savefig("docs/results/rl_learning_curves.png", dpi=120)
    Path("docs/results/rl_learning_curves.json").write_text(json.dumps(
        {"note": __doc__.strip(), "training_reward": curves}, indent=1) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["eval", "plot"])
    {"eval": evaluate, "plot": plot}[p.parse_args().action]()
