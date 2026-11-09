"""Control for replay.py: generate simulator logs with the same controllers, then
replay them with the identical procedure. Sim-on-sim error is the method's floor."""
import sys, os, math
sys.path[:0] = ['src', 'scripts', 'runs/obs_model']
import numpy as np
from pathlib import Path
import replay as RP
import real_state_starts as R
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from train_rig_rl import RIG

JITTER = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

def make_logs():
    logs = []
    for alg, ck, cfg in (('lqr', 'docs/results/sac_focused_policy.zip', 'configs/lqr_my_rig_coupling_delay.json'),
                         ('sac', 'docs/results/rig_rl_disturbance/sac_long/policy.zip', 'configs/sac_my_rig.json'),
                         ('sac', 'runs/sac_finetune/V1/policy.zip', 'configs/sac_my_rig.json')):
        c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
        env = stress_env(Path("configs/my_rig.json"), RIG); p = env.unwrapped; p.exact_hanging_reset_frac = 0.0
        if JITTER:
            from finetune_real import CommandNoise
            env = CommandNoise(env, (JITTER, JITTER))
        for sd in range(4):
            o, _ = env.reset(seed=900 + sd); c.reset(); S = []; A = []
            for k in range(750):
                f = o[-6:]; S.append([f[0], math.atan2(f[1], f[2]), f[3], f[4]])
                a = c.act(o); A.append(a)
                o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
                if te: break
            logs.append(('sac' if alg == 'sac' else 'lqr', np.array(S), np.array(A)))
    return logs

if __name__ == '__main__':
    logs = make_logs()
    RP.runs = lambda: list(range(len(logs)))
    R.load_run = lambda i: (logs[i][1], logs[i][2])
    # replay() decides 'sac' vs classical from the path; patch via a wrapper
    orig = RP.replay
    Eth, Ev, A, V0, kind = [], [], [], [], []
    import types
    for grp in ('lqr', 'sac'):
        idx = [i for i, l in enumerate(logs) if l[0] == grp]
        RP.runs = lambda idx=idx: [f'docs/x/{grp}/{i}' for i in idx]
        R.load_run = lambda r: (logs[int(r.split('/')[-1])][1], logs[int(r.split('/')[-1])][2])
        e1, e2, *_ = orig(RIG)
        print(f'[sim-on-sim {grp}] n={len(e1)} rms angle error by horizon (deg):', np.round(np.degrees(np.sqrt(np.mean(e1 ** 2, 0))), 2),
              ' velocity (rad/s):', np.round(np.sqrt(np.mean(e2 ** 2, 0)), 2), flush=True)
