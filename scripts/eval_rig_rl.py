"""Compare standalone RL policies on paired identified-rig simulations."""
import argparse
import hashlib
import json
from pathlib import Path

import transfer_stress
from pendulum_lab.robustness import CASES
from train_rig_rl import BASELINES, RIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47200)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--candidates", type=Path, help="Directory containing per-algorithm policy.zip files")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    params = Path("configs/my_rig.json")
    config = "configs/sac_my_rig.json"
    CASES.update(rig_exact=RIG, rig_varied={**RIG, "varied_start": True})
    policies = {name: (name, Path(path)) for name, (_, path, _) in BASELINES.items()}
    policies["sac_focused"] = ("sac", Path("docs/results/sac_focused_policy.zip"))
    if args.candidates:
        for name in BASELINES:
            policies[name + "_adapted"] = (name, args.candidates / name / "policy.zip")

    results = {}
    for label, (algorithm, path) in policies.items():
        print(label, path, flush=True)
        results[label] = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for case in ("rig_exact", "rig_varied", "combined"):
            results[label].update(transfer_stress.run(
                algorithm, config, params=params, checkpoint=path,
                episodes=1 if case == "rig_exact" else args.episodes, seed=args.seed, cases=(case,),
            ))
    with args.output.open("x") as report:
        json.dump({"params": str(params), "config": config, "seed": args.seed,
                   "cases": {name: CASES[name] for name in ("rig_exact", "rig_varied", "combined")},
                   "results": results}, report, indent=2)
        report.write("\n")


if __name__ == "__main__":
    main()
