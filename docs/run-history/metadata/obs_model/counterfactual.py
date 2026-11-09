"""On the same real observations right after capture, what would LQR's balance law
have commanded vs what the SAC policy commanded (and vice versa on LQR captures)?"""
import sys, glob, json, math
sys.path[:0] = ['src', 'scripts']
import numpy as np
from pathlib import Path
from pendulum_lab.controllers import load_controller
import real_state_starts as R
from pendulum_lab.contracts import state_from_obs
lqr = load_controller('lqr', Path('configs/my_rig.json'), Path('configs/lqr_my_rig_coupling_delay.json'), Path('docs/results/sac_focused_policy.zip'))
pols = {'v1': load_controller('sac', Path('configs/my_rig.json'), None, Path('runs/sac_finetune/V1/policy.zip')),
        'sac': load_controller('sac', Path('configs/my_rig.json'), None, Path('docs/results/rig_rl_disturbance/sac_long/policy.zip'))}
def lqr_cmd(obs):
    x = state_from_obs(obs)
    return float(np.clip(lqr.balance(x, obs, use_observer=False), -1, 1))   # arm reference 0, velocity blend on
def rows(runs, who):
    out = []
    for r in runs:
        d = np.load(r + '/episode_000.npz', allow_pickle=True); s = d['samples']; mode = d['hardware_timing'][:, 6]
        req = d['hardware_timing'][:, 8]
        states, acts = s[:, 1:5], s[:, 5]
        for k in range(4, len(s)):
            if mode[k] == 1 and mode[k - 1] != 1:
                for j in range(k, min(k + 15, len(s))):
                    if mode[j] != 1: break
                    obs = R.history(states, acts, j)
                    out.append((j - k, req[j] if np.isfinite(req[j]) else acts[j], lqr_cmd(obs),
                                {n: float(p.act(obs)) for n, p in pols.items()}, states[j]))
    return out
def report(name, data, actual_is):
    steps = np.array([d[0] for d in data]); actual = np.array([d[1] for d in data]); lq = np.array([d[2] for d in data])
    print(f'--- {name}: {len(data)} ticks in the first 0.3 s after capture')
    for n in pols:
        sc = np.array([d[3][n] for d in data])
        agree = np.mean(np.sign(sc) == np.sign(lq))
        big = np.abs(lq) > 0.3
        print(f'   {n:4s} vs LQR law: corr {np.corrcoef(sc, lq)[0,1]:+.2f}  sign agreement {agree:.0%} (when LQR |a|>0.3: {np.mean(np.sign(sc[big])==np.sign(lq[big])):.0%})  '
              f'mean|{n}| {np.mean(np.abs(sc)):.2f} vs mean|LQR| {np.mean(np.abs(lq)):.2f}; ratio fit {float(sc@lq/(lq@lq)):+.2f}')
lqr_runs = [r for r in sorted(glob.glob('docs/results/hardware/lqr/2026*')) if json.load(open(r + '/run.json'))['arguments'].get('config', '').endswith('lqr_my_rig_coupling_delay.json')]
report('LQR captures (real states LQR handled)', rows(lqr_runs, 'lqr'), 'lqr')
report('V1-balance captures', rows(sorted(glob.glob('runs/custom_v1raw_*')) + sorted(glob.glob('runs/custom_v1matched_*')), 'v1'), 'v1')
report('SAC-balance captures', rows(sorted(glob.glob('runs/custom_sacraw_*')), 'sac'), 'sac')
