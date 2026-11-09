"""Same analysis on rig logs and simulator logs: is the rig's pendulum-velocity reading
explained by the firmware's 8 ms backward window, or is there extra lag / noise?

For each sample, fit a local cubic to the (unwrapped) angle readings around it using
timestamps, then predict what the firmware window should report:
    v_pred = (theta_hat(t) - theta_hat(t - W - extra)) / W ... approximated by theta_hat'(t - tau)
Fit tau by least squares, report residual RMS by speed and correlation with acceleration.
Running the identical code on simulator logs (which use the firmware window model)
gives the expected numbers; the rig-minus-sim difference is the real discrepancy.
"""
import sys, os, glob, math, json
sys.path[:0] = ['src', 'scripts']
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

HALF = 3  # samples each side for local fit


def local_fit(t, th):
    """Return per-sample cubic coefficients (centered at t_k) for interior samples."""
    out = {}
    for k in range(HALF, len(t) - HALF):
        tt = t[k - HALF:k + HALF + 1] - t[k]
        c = np.polyfit(tt, th[k - HALF:k + HALF + 1], 3)
        out[k] = c
    return out


def analyse(t, th, v, true_v=None):
    th = np.unwrap(th)
    fits = local_fit(t, th)
    ks = np.array(sorted(fits))
    def pred(tau, window=None):
        p = []
        for k in ks:
            c = fits[k]
            if window:  # mean slope over [-tau-window, -tau]
                a, b = -tau - window, -tau
                p.append((np.polyval(c, b) - np.polyval(c, a)) / window)
            else:
                p.append(np.polyval(np.polyder(c), -tau))
        return np.array(p)
    vv = v[ks]
    acc = np.array([np.polyval(np.polyder(fits[k], 2), 0.0) for k in ks])
    res = {}
    # model A: firmware window exactly (8 ms ending at the stamp), extra lag e fitted
    best = None
    for e in np.arange(-0.004, 0.0121, 0.0005):
        r = vv - pred(e, window=0.008)
        m = np.abs(vv) > 2
        s = np.sqrt(np.mean(r[m] ** 2))
        if best is None or s < best[1]: best = (e, s)
    e = best[0]
    r = vv - pred(e, window=0.008)
    r0 = vv - pred(0.0, window=0.008)
    speed = np.abs(vv)
    for lo, hi in ((0, 2), (2, 8), (8, 15), (15, 40)):
        m = (speed >= lo) & (speed < hi)
        if m.sum() > 20:
            res[f'{lo}-{hi}'] = dict(n=int(m.sum()), rms_window_only=float(np.sqrt(np.mean(r0[m] ** 2))),
                                    rms_fitted=float(np.sqrt(np.mean(r[m] ** 2))),
                                    corr_acc=float(np.corrcoef(r0[m], acc[m])[0, 1]))
    out = dict(extra_lag_ms=e * 1e3, bins=res)
    if true_v is not None:
        tv = true_v[ks]; m = speed > 8
        out['true_reading_err_rms_fast'] = float(np.sqrt(np.mean((vv[m] - tv[m]) ** 2)))
    return out, (vv, r0, acc, speed)


def real_runs(pattern):
    T = []
    for r in sorted(glob.glob(pattern)):
        f = r + '/episode_000.npz'
        if not os.path.exists(f) or '163412' in r: continue
        d = np.load(f, allow_pickle=True)
        s, ht = d['samples'], d['hardware_timing']
        if len(s) < 100: continue
        t = (ht[:, 1] - ht[0, 1]) * 1e-6
        if np.any(np.diff(t) <= 0):
            t = np.unwrap(ht[:, 1], period=2**32)  # micros wrap
            t = (t - t[0]) * 1e-6
        T.append((t, s[:, 2], s[:, 4], None))
    return T


def sim_runs(ck, alg, cfg, seeds, noise=0.0):
    import torch; torch.set_num_threads(1)
    from pathlib import Path
    from pendulum_lab.robustness import stress_env, ObservationNoise
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), RIG); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    env = ObservationNoise(base, 0.0, noise) if noise else base
    T = []
    for sd in seeds:
        o, _ = env.reset(seed=sd); c.reset(); th = []; v = []; tv = []
        for k in range(750):
            a = c.act(o); o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            f = o[-6:]; th.append(math.atan2(f[1], f[2])); v.append(f[4])
            # truth at the same (stale) instant is not stored; use current qvel as rough truth
            tv.append(float(p.data.qvel[p._pen_qvel_addr]))
            if te: break
        T.append((np.arange(len(th)) * 0.02, np.array(th), np.array(v), np.array(tv)))
    return T


def pooled(T):
    allr = []
    outs = [analyse(*x) for x in T]
    lags = [o[0]['extra_lag_ms'] for o in outs]
    vv = np.concatenate([o[1][0] for o in outs]); r0 = np.concatenate([o[1][1] for o in outs])
    acc = np.concatenate([o[1][2] for o in outs]); sp = np.abs(vv)
    rows = {}
    for lo, hi in ((0, 2), (2, 8), (8, 15), (15, 40)):
        m = (sp >= lo) & (sp < hi)
        if m.sum() > 20:
            A = np.c_[acc[m], np.ones(m.sum())]
            coef, *_ = np.linalg.lstsq(A, r0[m], rcond=None)
            rr = r0[m] - A @ coef
            rows[f'{lo}-{hi}'] = dict(n=int(m.sum()), rms=float(np.sqrt(np.mean(r0[m] ** 2))),
                                     corr_acc=float(np.corrcoef(r0[m], acc[m])[0, 1]),
                                     lag_from_acc_ms=float(coef[0] * 1e3),
                                     rms_after_acc=float(np.sqrt(np.mean(rr ** 2))))
    tr = [o[0].get('true_reading_err_rms_fast') for o in outs if 'true_reading_err_rms_fast' in o[0]]
    return dict(median_extra_lag_ms=float(np.median(lags)), bins=rows,
                true_err_fast=float(np.mean(tr)) if tr else None)


if __name__ == '__main__':
    from multiprocessing import Pool
    which = sys.argv[1]
    if which == 'real':
        for pat in ('runs/sac_*_2026*', 'docs/results/hardware/sac/2026*', 'docs/results/hardware/lqr/2026*'):
            print(pat, json.dumps(pooled(real_runs(pat)), indent=1))
    else:
        jobs = [('docs/results/rig_rl_disturbance/sac_long/policy.zip', 'sac', 'configs/sac_my_rig.json', range(10)),
                ('runs/sac_finetune/R50/policy.zip', 'sac', 'configs/sac_my_rig.json', range(10)),
                ('docs/results/sac_focused_policy.zip', 'lqr', 'configs/lqr_my_rig_coupling_delay.json', range(10))]
        with Pool(3) as pool:
            Ts = pool.starmap(sim_runs, jobs)
        for j, T in zip(jobs, Ts):
            print('SIM', j[0], json.dumps(pooled(T), indent=1))
