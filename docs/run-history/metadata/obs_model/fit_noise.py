"""Fit rig reading jitter (angle and 8 ms velocity) against the commands just sent.
Fit on hardware runs before V1; V1 runs are held out."""
import sys, os, glob, json
sys.path[:0] = ['runs/obs_model', 'src', 'scripts']
import numpy as np
import velocity_check as V

def residuals(r):
    d = np.load(r + '/episode_000.npz', allow_pickle=True); s, ht = d['samples'], d['hardware_timing']
    if len(s) < 100: return None
    t = (np.unwrap(ht[:, 1].astype(float), period=2**32) - ht[0, 1]) * 1e-6
    th = np.unwrap(s[:, 2]); f = V.local_fit(t, th); ks = np.array(sorted(f))
    ks = ks[ks >= 2]
    vr = s[ks, 4] - np.array([(np.polyval(f[k], 0) - np.polyval(f[k], -0.008)) / 0.008 for k in ks])
    ar = np.array([th[k] - np.polyval(f[k], 0) for k in ks])
    a = s[:, 5]
    X = np.c_[np.ones(len(ks)), np.abs(a[ks - 1]), np.abs(a[ks - 1] - a[ks - 2]), np.abs(s[ks, 4])]
    return X, vr, ar

def runs(held):
    out = []
    for r in sorted(glob.glob('docs/results/hardware/*/2026*')) + sorted(glob.glob('runs/*_2026*')):
        if not os.path.exists(r + '/episode_000.npz') or '163412' in r: continue
        if ('sac_V1' in r) != held: continue
        if any(x in r for x in ('_rig_disturbance', 'coupling_delay', 'transfer', 'lqr_match', 'strict', 'low_r', 'mpc_hw', 'lqr_calibrated', 'lqr_swing', 'before_')): continue
        x = residuals(r)
        if x: out.append(x)
    return [np.concatenate(c) for c in zip(*out)]

if __name__ == '__main__':
    X, vr, ar = runs(held=False)
    # model E|res| = X @ c  (mean absolute value; rms derived assuming the fitted tail)
    cv, *_ = np.linalg.lstsq(X, np.abs(vr), rcond=None)
    ca, *_ = np.linalg.lstsq(X, np.abs(ar), rcond=None)
    # tail shape: standardized residual distribution
    zv = vr / np.maximum(X @ cv, 1e-3); za = ar / np.maximum(X @ ca, 1e-3)
    def tdf(z):
        k = np.mean(z**4) / np.mean(z**2)**2
        return k
    lag1v = np.corrcoef(vr[:-1], vr[1:])[0, 1]; lag1a = np.corrcoef(ar[:-1], ar[1:])[0, 1]
    corr_va = np.corrcoef(vr, ar)[0, 1]
    names = ['const', '|a_prev|', '|da_prev|', '|pen_vel|']
    print('velocity mean|res| =', dict(zip(names, np.round(cv, 3))), 'kurtosis of standardized', round(tdf(zv), 1), 'lag1', round(lag1v, 2))
    print('angle   mean|res| rad =', dict(zip(names, np.round(ca, 5))), 'kurtosis', round(tdf(za), 1), 'lag1', round(lag1a, 2))
    print('corr(vel res, angle res)', round(corr_va, 2))
    Xh, vrh, arh = runs(held=True)
    for nm, c, y in (('velocity', cv, vrh), ('angle', ca, arh)):
        pred = Xh @ c
        print(f'held-out V1 {nm}: predicted mean|res| {pred.mean():.4f} actual {np.abs(y).mean():.4f}; corr(pred,|res|) {np.corrcoef(pred, np.abs(y))[0,1]:.2f}')
    json.dump({'velocity_mean_abs': cv.tolist(), 'angle_mean_abs': ca.tolist(), 'features': names,
               'velocity_kurtosis': float(tdf(zv)), 'angle_kurtosis': float(tdf(za)),
               'lag1_velocity': float(lag1v), 'lag1_angle': float(lag1a), 'corr_vel_angle': float(corr_va)},
              open('runs/obs_model/noise_fit.json', 'w'), indent=1)
