"""Closed-loop identification near upright: regress pendulum angular acceleration on
[sin theta, pendulum speed, arm speed, commands at lags 0..4] from LQR holds, on the rig and in the
simulator (same pipeline, with the measured jitter). Compare coefficients."""
import sys, glob, json, math
sys.path[:0] = ['src', 'scripts', 'runs/obs_model', 'runs/sac_finetune']
import numpy as np
from pathlib import Path
import velocity_check as V
LAGS = 5
def features(t, s):
    th = np.unwrap(s[:, 1]); f = V.local_fit(t, th)
    rows, y = [], []
    a = s[:, 4]
    for k, c in f.items():
        if k < LAGS or abs(s[k, 1]) > math.radians(12): continue
        acc = np.polyval(np.polyder(c, 2), 0); w = np.polyval(np.polyder(c), 0)
        rows.append([math.sin(s[k, 1]), w, s[k, 2], *[a[k - 1 - L] for L in range(LAGS)]]); y.append(acc)
    return rows, y
def fit(X, y):
    X = np.asarray(X); y = np.asarray(y)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.var(y - X @ c) / np.var(y)
    # bootstrap by blocks of rows for rough CIs
    rng = np.random.default_rng(0); B = []
    n = len(y); blk = 50
    for _ in range(200):
        idx = np.concatenate([np.arange(i, min(i + blk, n)) for i in rng.integers(0, n, n // blk)])
        cb, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None); B.append(cb)
    return c, np.std(B, 0), r2, n
names = ['sin(theta)', 'pen speed', 'arm speed'] + [f'u[k-{L+1}]' for L in range(LAGS)]
def show(label, X, y):
    c, sd, r2, n = fit(X, y)
    print(f'{label} (n={n}, R2={r2:.2f})')
    print('   ' + '  '.join(f'{nm} {ci:+7.1f}±{s:4.1f}' for nm, ci, s in zip(names, c, sd)))
    return c
# rig LQR holds
X, y = [], []
for r in sorted(glob.glob('docs/results/hardware/lqr/2026*')):
    if not json.load(open(r + '/run.json'))['arguments'].get('config', '').endswith('lqr_my_rig_coupling_delay.json'): continue
    d = np.load(r + '/episode_000.npz', allow_pickle=True); s = d['samples']
    if len(s) < 60 or len(d['hardware_timing']) != len(s): continue
    mode = d['hardware_timing'][:, 6]
    t = (np.unwrap(d['hardware_timing'][:, 1].astype(float), period=2**32) - d['hardware_timing'][0, 1]) * 1e-6
    st = np.c_[s[:, 1:5], s[:, 5]]
    # only stretches inside a hold
    m = mode == 1
    idx = np.where(m)[0]
    if len(idx) < 30: continue
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(seg) < 30: continue
        lo, hi = seg[0], seg[-1] + 1
        rx, ry = features(t[lo:hi], st[lo:hi][:, [0, 1, 2, 3, 4]][:, [0, 1, 2, 3, 4]])
        X += rx; y += ry
cr = show('RIG  LQR holds', X, y)
# simulator LQR holds, same pipeline, with/without measured jitter and kick
from pendulum_lab.controllers import load_controller
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
from finetune_real import CommandNoise
for kick, jit in ((0.0, False), (0.2, True)):
    c = load_controller('lqr', Path('configs/my_rig.json'), Path('configs/lqr_my_rig_coupling_delay.json'), Path('docs/results/sac_focused_policy.zip'))
    base = stress_env(Path('configs/my_rig.json'), RIG); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    p._accel_transport.zero_cross_kick_rad_s = kick
    env = CommandNoise(base, (1, 1)) if jit else base
    X, y = [], []
    for sd in range(10):
        o, _ = env.reset(seed=7700 + sd); c.reset(); S = []; on = []
        for k in range(750):
            a = c.act(o); f = o[-6:]
            S.append([f[0], math.atan2(f[1], f[2]), f[3], f[4], a]); on.append(c.balancing)
            o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            if te: break
        S = np.array(S); on = np.array(on); idx = np.where(on)[0]
        if len(idx) < 30: continue
        # commands are logged as sent at k; features() uses a[k-1-L], so shift by one to match rig logs (sent action at row k)
        rx, ry = features(np.arange(len(idx)) * 0.02, S[idx]); X += rx; y += ry
    show(f'SIM  LQR holds kick {kick} jitter {int(jit)}', X, y)
