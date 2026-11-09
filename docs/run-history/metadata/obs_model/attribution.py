"""Which observation parts make SAC disagree with the LQR law on real post-capture states?
Clean one part at a time (from smoothed signals) and measure agreement with the LQR law."""
import sys, glob, json, math
sys.path[:0] = ['src', 'scripts', 'runs/obs_model']
import numpy as np
from pathlib import Path
src = open('runs/obs_model/counterfactual.py').read().split("lqr_runs =")[0]
ns = {}; exec(src, ns)
import velocity_check as V
from pendulum_lab.contracts import observation_frame

def smooth_run(s):
    t = np.arange(len(s)) * 0.02
    th = np.unwrap(s[:, 2]); fa = V.local_fit(t, th)
    arm = s[:, 1]; fm = V.local_fit(t, arm)
    S = s[:, 1:5].copy()
    for k in fa:
        S[k, 1] = math.remainder(np.polyval(fa[k], 0), 2 * math.pi); S[k, 3] = np.polyval(np.polyder(fa[k]), 0)
        S[k, 2] = np.polyval(np.polyder(fm[k]), 0)
    return S, set(fa)

def obs_from(states, acts, k, parts, clean, own_actions=None):
    frames = []
    for j in range(k - 3, k + 1):
        st = states[j].copy()
        for i, name in ((1, 'angle'), (2, 'arm_speed'), (3, 'pen_speed')):
            if name in parts: st[i] = clean[j, i]
        prev = acts[j - 1]
        if 'actions' in parts and own_actions is not None: prev = own_actions[j - 1]
        frames.append(observation_frame(st, prev))
    return np.concatenate(frames)

lqr_runs = [r for r in sorted(glob.glob('docs/results/hardware/lqr/2026*')) if json.load(open(r + '/run.json'))['arguments'].get('config', '').endswith('lqr_my_rig_coupling_delay.json')]
groups = {'LQR captures': lqr_runs, 'SAC-balance captures': sorted(glob.glob('runs/custom_sacraw_*'))}
variants = [(), ('pen_speed',), ('angle',), ('arm_speed',), ('angle', 'pen_speed'), ('angle', 'pen_speed', 'arm_speed'), ('actions',), ('angle', 'pen_speed', 'arm_speed', 'actions')]
pol = ns['pols']['sac']
for gname, runs in groups.items():
    res = {v: ([], []) for v in variants}
    for r in runs:
        d = np.load(r + '/episode_000.npz', allow_pickle=True); s = d['samples']; mode = d['hardware_timing'][:, 6]
        states, acts = s[:, 1:5], s[:, 5]
        clean, valid = smooth_run(s)
        # 'actions': what SAC itself would have sent at the previous ticks (its own history)
        own = acts.copy()
        for k in range(4, len(s)):
            if mode[k] == 1 and mode[k - 1] != 1:
                for j in range(k, min(k + 15, len(s))):
                    if mode[j] != 1 or not all(i in valid for i in range(j - 3, j + 1)): break
                    target = ns['lqr_cmd'](obs_from(states, acts, j, ('angle', 'pen_speed', 'arm_speed'), clean))
                    for v in variants:
                        o = obs_from(states, acts, j, v, clean, own)
                        res[v][0].append(float(pol.act(o))); res[v][1].append(target)
                    own[j] = res[('actions',)][0][-1]
    print(f'--- {gname}')
    for v, (a, b) in res.items():
        a, b = np.array(a), np.array(b)
        print(f'   cleaned {"+".join(v) or "nothing":40s} SAC vs LQR law: corr {np.corrcoef(a, b)[0,1]:+.2f}  sign {np.mean(np.sign(a)==np.sign(b)):.0%}  (n={len(a)})')
