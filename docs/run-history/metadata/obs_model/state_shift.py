import sys, glob, json, math
sys.path[:0] = ['src', 'scripts', 'runs/sac_finetune']
import numpy as np
from pathlib import Path
src = open('runs/obs_model/counterfactual.py').read().split("lqr_runs =")[0]
ns = {}; exec(src, ns)
import real_state_starts as R
from pendulum_lab.controllers import load_controller
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
pol = ns['pols']['sac']
def collect_real():
    runs = [r for r in sorted(glob.glob('docs/results/hardware/lqr/2026*')) if json.load(open(r + '/run.json'))['arguments'].get('config', '').endswith('lqr_my_rig_coupling_delay.json')]
    O = []
    for r in runs:
        d = np.load(r + '/episode_000.npz', allow_pickle=True); s = d['samples']; mode = d['hardware_timing'][:, 6]
        for k in range(4, len(s)):
            if mode[k] == 1 and mode[k - 1] != 1:
                for j in range(k, min(k + 15, len(s))):
                    if mode[j] != 1: break
                    O.append(R.history(s[:, 1:5], s[:, 5], j))
    return np.array(O)
def collect_sim():
    c = load_controller('lqr', Path('configs/my_rig.json'), Path('configs/lqr_my_rig_coupling_delay.json'), Path('docs/results/sac_focused_policy.zip'))
    env = stress_env(Path('configs/my_rig.json'), RIG); p = env.unwrapped; p.exact_hanging_reset_frac = 0.0
    O = []
    for sd in range(40):
        o, _ = env.reset(seed=7500 + sd); c.reset(); start = None
        for k in range(400):
            a = c.act(o)
            if c.balancing and start is None: start = k
            if start is not None and k - start < 15: O.append(o.copy())
            o, *_ = env.step(np.array([a], dtype=np.float32))
    return np.array(O)
for name, O in (('sim LQR captures', collect_sim()), ('rig LQR captures', collect_real())):
    f = O[:, -6:]; th = np.degrees(np.arctan2(f[:, 1], f[:, 2]))
    sc = np.array([pol.act(o) for o in O]); lq = np.array([ns['lqr_cmd'](o) for o in O])
    prev = O[:, 5::6]
    print(f'{name}: n={len(O)} |arm| median {np.degrees(np.median(np.abs(f[:,0]))):4.1f}° (p90 {np.degrees(np.percentile(np.abs(f[:,0]),90)):4.1f}) |arm vel| {np.median(np.abs(f[:,3])):.2f} |theta| {np.median(np.abs(th)):4.1f}° |pen vel| {np.median(np.abs(f[:,4])):.2f}  |prev a| {np.median(np.abs(prev)):.2f} action-history sign changes {np.mean(np.sign(prev[:,1:])!=np.sign(prev[:,:-1])):.0%}')
    print(f'    SAC vs LQR law: corr {np.corrcoef(sc, lq)[0,1]:+.2f} sign {np.mean(np.sign(sc)==np.sign(lq)):.0%}')
    dis = np.sign(sc) != np.sign(lq)
    for feat, x in (('|arm|deg', np.degrees(np.abs(f[:, 0]))), ('|arm vel|', np.abs(f[:, 3])), ('|theta|deg', np.abs(th)), ('|pen vel|', np.abs(f[:, 4]))):
        q = np.percentile(x, [0, 50, 100]); lo = x <= q[1]
        print(f'      disagreement when {feat} <= {q[1]:.2f}: {np.mean(dis[lo]):.0%}   above: {np.mean(dis[~lo]):.0%}')
