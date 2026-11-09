"""Score checkpoints: chatter from real upright states, hanging-start passes over delay, standard cases."""
import glob, json, os, sys, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path[:0] = ["src", "scripts", "runs/sac_real_starts"]
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REAL = [r for r in sorted(glob.glob("docs/results/hardware/sac/2026*")) + sorted(glob.glob("runs/sac_A_*")) if "163412" not in r]


def real_chatter(ck, delay=0.030):
    from pendulum_lab.robustness import stress_env
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    import real_state_starts as R
    c = load_controller("sac", Path("configs/my_rig.json"), Path("configs/sac_my_rig.json"), Path(ck))
    env = stress_env(Path("configs/my_rig.json"), {**RIG, "delay_s": delay}); env.reset(seed=0); p = env.unwrapped
    sat, sw, held = [], [], []
    for r in REAL:
        s, a = R.load_run(r)
        for k in [k for k in R.near_upright(s, theta_max=0.2, last=60) if abs(s[k, 3]) < 2][::3]:
            o = R.set_state(env, s, a, k); c.reset(); acts = []; ok = True
            for j in range(50):
                u = c.act(o); acts.append(u); o, _, te, _, _ = env.step(np.array([u], dtype=np.float32))
                th = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2 * math.pi)
                if te or abs(th) > math.pi / 2: ok = False; break
            acts = np.array(acts)
            sat.append(np.mean(np.abs(acts[:20]) > 0.95)); sw.append(np.mean(np.abs(np.diff(acts[:20])) > 1.0)); held.append(ok)
    return {"full_strength": float(np.mean(sat)), "big_swings": float(np.mean(sw)), "held_1s": float(np.mean(held)), "n": len(sat)}


def job(arg):
    import torch; torch.set_num_threads(1)
    ck, kind, d = arg
    if kind == "real":
        return ck, kind, d, real_chatter(ck, d)
    from drift_test import run
    o = run(ck, 0.03, d, eps=6)
    return ck, kind, d, {"pass": sum(r["success"] for r, _ in o), "full_strength": float(np.nanmean([s for _, s in o]))}


if __name__ == "__main__":
    cks = sys.argv[1:]
    jobs = [(c, "real", 0.030) for c in cks] + [(c, "hang", d) for c in cks for d in (0.030, 0.036, 0.042)]
    with Pool(min(14, len(jobs))) as pool:
        res = pool.map(job, jobs)
    table = {}
    for ck, kind, d, r in res:
        table.setdefault(ck, {})[f"{kind}_{int(d*1e3)}"] = r
    for ck, t in table.items():
        r = t["real_30"]
        h = " ".join(f"{d}ms:{t[f'hang_{d}']['pass']}/6({t[f'hang_{d}']['full_strength']:.0%})" for d in (30, 36, 42))
        print(f"{ck[-40:]:40s} real states: full {r['full_strength']:.0%} swings {r['big_swings']:.0%} held1s {r['held_1s']:.0%} | from hanging pass(full-strength) {h}")
    Path("runs/sac_real_starts/eval_all.json").write_text(json.dumps(table, indent=2))
