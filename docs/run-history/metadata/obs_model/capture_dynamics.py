"""How captures end: first 0.4 s after capture for LQR vs SAC balance on the rig."""
import glob, json, os, math
import numpy as np
def captures(run):
    d = np.load(run + '/episode_000.npz', allow_pickle=True); s = d['samples']; mode = d['hardware_timing'][:, 6]
    out = []; start = None
    for k, m in enumerate(mode):
        if m == 1 and start is None: start = k
        if (m != 1 or k == len(mode) - 1) and start is not None:
            out.append((start, k, s)); start = None
    return out
def describe(name, runs):
    rows = []
    for r in runs:
        for k0, k1, s in captures(r):
            arm, th, av, pv, a = s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
            w = slice(k0, min(k0 + 20, k1))
            n = k1 - k0
            sgn = np.sign(th[k0:k1]); flips = int(np.sum(sgn[1:] != sgn[:-1]))
            rows.append(dict(dur=n / 50, th0=abs(math.degrees(th[k0])), w0=abs(pv[k0]), a_mean=np.mean(np.abs(a[w])),
                             sat=np.mean(np.abs(a[w]) > 0.95), jumps=np.mean(np.abs(np.diff(a[w])) > 0.5) if n > 2 else 0,
                             flips_per_s=flips / max(n / 50, 0.02), exit_arm=abs(math.degrees(arm[k1 - 1])),
                             same_sign_exit=bool(np.sign(th[k1 - 1]) == np.sign(a[max(k1 - 3, k0):k1]).mean().round()) ))
    R = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    short = R['dur'] < 1
    print(f"{name:12s} n={len(rows):2d} entry |theta| {np.median(R['th0']):4.1f}° |w| {np.median(R['w0']):4.2f}  "
          f"first 0.4 s: mean|a| {np.mean(R['a_mean']):.2f} full-strength {np.mean(R['sat']):.0%} jumps {np.mean(R['jumps']):.0%}  "
          f"theta sign flips {np.median(R['flips_per_s']):4.1f}/s  exit |arm| {np.median(R['exit_arm']):3.0f}°")
    return rows
lqr = [r for r in sorted(glob.glob('docs/results/hardware/lqr/2026*')) if json.load(open(r + '/run.json'))['arguments'].get('config', '').endswith('lqr_my_rig_coupling_delay.json')]
describe('LQR', lqr)
for g in ('v1raw', 'v1matched', 'sacraw'):
    describe(g, sorted(glob.glob(f'runs/custom_{g}_*')))
