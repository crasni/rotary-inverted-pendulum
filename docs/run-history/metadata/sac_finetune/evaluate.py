"""Score fine-tuned SAC checkpoints: plain-sim task, smoothness, agreement with LQR's real balancing."""
import glob, json, math, os, re, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path[:0] = ["src", "scripts"]
from multiprocessing import Pool
from pathlib import Path
import numpy as np


def held_lqr_states():
    """Near-upright moments from held-out LQR runs while LQR was holding (from the training.json split)."""
    meta = json.loads(Path("runs/sac_finetune/R50/training.json").read_text()) if Path("runs/sac_finetune/R50/training.json").exists() else None
    import finetune_real as F
    runs = F.default_runs()
    held = runs[::5]
    out_obs, out_act = [], []
    import real_state_starts as R
    for r in held:
        if "/lqr/" not in r:
            continue
        s, a = R.load_run(r)
        for k in range(4, len(s) - 50):
            if np.all(np.abs(s[k:k + 50, 1]) < math.radians(20)):   # LQR held the next second
                out_obs.append(R.history(s, a, k)); out_act.append(a[k])
    return np.array(out_obs, dtype=np.float32), np.array(out_act)


def job(ck):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.controllers import load_controller
    from pendulum_lab.robustness import stress_env
    from train_rig_rl import RIG
    c = load_controller("sac", Path("configs/my_rig.json"), Path("configs/sac_my_rig.json"), Path(ck))
    res = {}
    for label, delay in (("sim18", 0.018), ("sim10", 0.010)):
        env = stress_env(Path("configs/my_rig.json"), {**RIG, "delay_s": delay}); p = env.unwrapped; p.exact_hanging_reset_frac = 0.0
        ok = 0; jumps = []
        for e in range(8):
            o, _ = env.reset(seed=700 + e); c.reset(); th = []; w = []; prev = 0.0
            for k in range(750):
                a = c.act(o); o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
                t = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2 * math.pi); th.append(t); w.append(float(p.data.qvel[p._pen_qvel_addr]))
                if abs(t) < 0.35: jumps.append(abs(a - prev) > 0.5)
                prev = a
                if te: break
            th = np.array(th); w = np.array(w)
            ok += len(th) == 750 and bool(np.all(np.abs(th[-500:]) < math.radians(15)) and np.all(np.abs(w[-500:]) < 4))
        res[label] = ok; res[label + "_jumps"] = float(np.mean(jumps)) if jumps else float("nan")
    obs, act = held_lqr_states()
    pol = np.array([c.controller.act(x) for x in obs])
    res["lqr_agree_corr"] = float(np.corrcoef(pol, act)[0, 1]); res["lqr_abs_diff"] = float(np.mean(np.abs(pol - act)))
    res["n_lqr_states"] = len(obs)
    return ck, res


if __name__ == "__main__":
    cks = ["docs/results/rig_rl_disturbance/sac_long/policy.zip"]
    for v in sys.argv[1:]:
        files = sorted(glob.glob(f"runs/sac_finetune/{v}/checkpoints/*.zip"), key=lambda f: int(re.findall(r"(\d+)_steps", f)[0]))
        cks += files
    with Pool(min(16, len(cks))) as pool:
        for ck, r in pool.map(job, cks):
            print(f"{ck[-40:]:40s} sim pass 18ms {r['sim18']}/8 10ms {r['sim10']}/8 | jumps upright {r['sim18_jumps']:.0%} | vs LQR real balancing: corr {r['lqr_agree_corr']:+.2f}, mean |diff| {r['lqr_abs_diff']:.2f} (n={r['n_lqr_states']})", flush=True)
