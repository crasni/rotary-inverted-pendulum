import sys, math; sys.path[:0]=['src','scripts','runs/sac_finetune']
import numpy as np
from pathlib import Path
from pendulum_lab.controllers import load_controller
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
from finetune_real import CommandNoise
import importlib.util
spec = importlib.util.spec_from_file_location('cf', 'runs/obs_model/counterfactual.py')
src = open('runs/obs_model/counterfactual.py').read().split("lqr_runs =")[0]
ns = {}; exec(src, ns)
for name, cfg, pol in (('V1 balancing', 'configs/custom_sacbal_v1_raw.json', 'v1'), ('SAC balancing', 'configs/custom_sacbal_sac_raw.json', 'sac')):
    for kick, jit in ((0.0, False), (0.2, True)):
        c = load_controller('custom', Path('configs/my_rig.json'), Path(cfg), Path('docs/results/sac_focused_policy.zip'))
        base = stress_env(Path('configs/my_rig.json'), RIG); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
        p._accel_transport.zero_cross_kick_rad_s = kick
        env = CommandNoise(base, (1, 1)) if jit else base
        sc, lq = [], []
        for sd in range(12):
            o, _ = env.reset(seed=7300 + sd); c.reset(); start = None
            for k in range(300):
                was = c.balancing
                a = c.act(o)
                if c.balancing and start is None: start = k
                if start is not None and k - start < 15 and c.balancing:
                    sc.append(ns['pols'][pol].act(o)); lq.append(ns['lqr_cmd'](o))
                o, *_ = env.step(np.array([a], dtype=np.float32))
        sc, lq = np.array(sc), np.array(lq)
        print(f'sim {name:14s} kick {kick} jitter {int(jit)}: first 0.3 s after capture, {pol} vs LQR law corr {np.corrcoef(sc, lq)[0,1]:+.2f} sign agree {np.mean(np.sign(sc)==np.sign(lq)):.0%} ratio {float(sc@lq/(lq@lq)):+.2f}  (n={len(sc)})', flush=True)
