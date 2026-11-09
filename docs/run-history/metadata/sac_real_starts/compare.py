"""Score SAC checkpoints on held-out real states and the existing simulator cases."""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path[:0] = ["src", "scripts"]

HELD_OUT = ["docs/results/hardware/sac/20261107_163427", "docs/results/hardware/sac/20261107_165745"]
STEPS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["100000", "200000", "300000", "400000"]


def score(path):
    import torch
    torch.set_num_threads(1)
    import real_state_starts, transfer_stress
    from pendulum_lab.robustness import CASES
    from train_rig_rl import RIG
    CASES.update(rig_exact=RIG, rig_varied={**RIG, "varied_start": True},
                 rig_delay30={**RIG, "delay_s": 0.030, "varied_start": True})
    out = {"real_heldout": real_state_starts.test("sac", Path(path), [Path(p) for p in HELD_OUT])}
    for case, n in (("rig_exact", 1), ("rig_varied", 10), ("combined", 10), ("rig_delay30", 10)):
        out[case] = transfer_stress.run("sac", "configs/sac_my_rig.json", params=Path("configs/my_rig.json"),
                                        checkpoint=Path(path), episodes=n, seed=47200, cases=(case,))[case]["passed"]
    return path, out


if __name__ == "__main__":
    paths = ["docs/results/rig_rl_disturbance/sac_long/policy.zip"]
    for v in "ABC":
        for s in STEPS:
            p = Path(f"runs/sac_real_starts/{v}/checkpoints/sac_{1675000 + int(s)}_steps.zip")
            if p.exists():
                paths.append(str(p))
    with Pool(min(14, len(paths))) as pool:
        results = dict(pool.map(score, paths))
    Path("runs/sac_real_starts/compare.json").write_text(json.dumps(results, indent=2))
    print(f"{'checkpoint':58s} realHeld fullStr exact varied/10 combined/10 delay30/10")
    for p, r in results.items():
        h = r["real_heldout"]
        print(f"{p:58s} {h['sim_held']:7.0%} {h['full_strength']:7.1%} {r['rig_exact']:5d} {r['rig_varied']:9d} {r['combined']:11d} {r['rig_delay30']:10d}")
