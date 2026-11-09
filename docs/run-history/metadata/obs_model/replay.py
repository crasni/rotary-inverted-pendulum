"""Open-loop replay near upright: start the simulator at a real state, feed the logged
commands, and compare its readings with the rig's over the next 6 samples (120 ms).
Reports error growth and whether the error is explained by the commands sent
(i.e. the pendulum responds to the arm more/less strongly than simulated)."""
import sys, os, glob, math, json
sys.path[:0] = ['src', 'scripts', 'runs/obs_model']
import numpy as np
from pathlib import Path
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R

H = 6
HELD = ('sac_V1',)
SPLIT = 'all'
def runs():
    out = []
    for r in sorted(glob.glob('docs/results/hardware/*/2026*')) + sorted(glob.glob('runs/sac_*_2026*')):
        if not os.path.exists(r + '/episode_000.npz') or '163412' in r: continue
        if any(x in r for x in ('_rig_disturbance', 'coupling_delay', 'transfer', 'lqr_match', 'strict', 'low_r')): continue
        held = any(h in r for h in HELD)
        if SPLIT == 'all' or (SPLIT == 'held') == held:
            out.append(r)
    return out

def smooth(s):
    """Replace pendulum angle/velocity with a local cubic fit of the angle readings."""
    import velocity_check as V
    th = np.unwrap(s[:, 1]); t = np.arange(len(s)) * 0.02
    out = s.copy()
    f = V.local_fit(t, th)
    for k, c in f.items():
        out[k, 1] = math.remainder(np.polyval(c, 0.0), 2 * math.pi); out[k, 3] = np.polyval(np.polyder(c), 0.0)
    return out, set(f)


def set_actuator(p, kp, zeta):
    I = 1.065625e-4
    p.model.actuator_gainprm[0, 0] = kp
    p.model.actuator_biasprm[0, 1] = -kp
    p.model.actuator_biasprm[0, 2] = -2 * zeta * math.sqrt(kp * I)


def replay(rig, coupling=1.0, max_accel=None, sel="all", smoothed=True, kp=None, zeta=1.0, kick=0.0):
    env = stress_env(Path('configs/my_rig.json'), rig); env.reset(seed=0); p = env.unwrapped
    p._coupling_gain = coupling
    if max_accel: p.max_accel_rad_s2 *= max_accel
    if kp: set_actuator(p, kp, zeta)
    p._accel_transport.zero_cross_kick_rad_s = kick
    global EARM
    E_th, E_v, A, V0, kind = [], [], [], [], []
    EARM = []
    for r in runs():
        s, a = R.load_run(r)
        if smoothed:
            s, valid = smooth(s)
        alg = r.split('/')[-2] if 'docs' in r else 'sac'
        if sel != 'all' and (alg == 'sac') != (sel == 'sac'): continue
        for k in range(5, len(s) - H - 1, 4):
            if not (abs(s[k, 1]) < math.radians(25) and abs(s[k, 3]) < 6): continue
            if smoothed and not all(j in valid for j in range(k, k + H + 1)): continue
            R.set_state(env, s, a, k)
            p._coupling_gain = coupling
            # The command sent at k-1 is already active on the real stepper (18 ms
            # transport); the 8 ms velocity window lags the true arm speed by ~4 ms.
            # Commands sent at k-j (j=1..3) activate at delay - 20j ms relative to k:
            # the latest already-active one is current, later ones stay queued.
            lim = p._motor_max_accel_rad_s2
            cmd = lambda j: float(np.clip(a[k - j] * p.max_accel_rad_s2, -lim, lim))
            d = p._transport_delay_s
            act = [(d - 0.02 * j, cmd(j)) for j in (3, 2, 1)]
            active = [x for x in act if x[0] <= 0]; pending = [x for x in act if x[0] > 0]
            acc = active[-1][1] if active else 0.0
            p._accel_transport.reset(position_rad=float(s[k, 0]), velocity_rad_s=float(s[k, 2]) + acc * 0.004,
                                     active_accel_rad_s2=acc)
            for tq, aq in pending:
                p._accel_transport._queue.append((tq, aq))
            eth, ev, earm = [], [], []
            for h in range(H):
                o, *_ = env.step(np.array([a[k + h]], dtype=np.float32))
                f = o[-6:]; th_sim = math.atan2(f[1], f[2])
                if smoothed:  # compare simulator truth with the smoothed rig angle
                    th_sim = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2 * math.pi)
                    f = f.copy(); f[4] = float(p.data.qvel[p._pen_qvel_addr])
                eth.append(math.remainder(s[k + h + 1, 1] - th_sim, 2 * math.pi)); ev.append(s[k + h + 1, 3] - f[4])
                earm.append(s[k + h + 1, 0] - float(o[-6]))
            EARM.append(earm); E_th.append(eth); E_v.append(ev); A.append(a[k - 2:k + H]); V0.append(s[k]); kind.append(alg)
    return np.array(E_th), np.array(E_v), np.array(A), np.array(V0), np.array(kind)

if __name__ == '__main__':
    Eth, Ev, A, V0, kind = replay(RIG)
    print(f'{len(Eth)} windows near upright')
    for grp in ('lqr', 'sac'):
        m = kind == grp if grp == 'sac' else kind != 'sac'
        print(f'[{"classical" if grp=="lqr" else "SAC"}] n={m.sum()}  rms angle error by horizon (deg):',
              np.round(np.degrees(np.sqrt(np.mean(Eth[m] ** 2, 0))), 2), ' velocity (rad/s):', np.round(np.sqrt(np.mean(Ev[m] ** 2, 0)), 2))
    # how much of the h=3 velocity error is a linear function of the commands sent?
    for h in (1, 2, 3, 5):
        y = Ev[:, h]
        X = np.c_[A, np.ones(len(A))]
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.var(y - X @ c) / np.var(y)
        print(f'h={h+1}: velocity error explained by command history R^2={r2:.2f}; coefs (a[k-2..k+{H-1}]):', np.round(c[:-1], 2))
    # sign/magnitude: does the rig respond more to commands? compare error to sim's own command response
    np.savez('runs/obs_model/replay_warm.npz', Eth=Eth, Ev=Ev, A=A, V0=V0, kind=kind)
