"""Fit a kinematic model of the real stepper's step count to driven captures:
delayed piecewise-constant acceleration (gain g, delay d) plus a velocity kick of
size k in the acceleration's direction whenever the velocity crosses zero."""
import math, itertools, sys
import numpy as np

def load(f):
    d = np.load(f, allow_pickle=True); r = d['samples']; t0 = float(d['start_host_time_s'])
    ct = d['command_t'] - (t0 if d['command_t'][0] > 1000 else 0)
    return r[:, 0] - t0, r[:, 3], ct, d['accel_commands']

def simulate(t, ct, ca, d, g, k, vmax=5.0, dt=2e-4):
    T = np.arange(0, t[-1] + dt, dt); pos = np.zeros_like(T); x = v = 0.0
    act = ct + d; j = -1; a = 0.0
    for i, tt in enumerate(T):
        while j + 1 < len(act) and act[j + 1] <= tt:
            j += 1; a = g * ca[j]
        v_new = v + a * dt
        if v != 0 and np.sign(v_new) != np.sign(v) and a != 0 or (v == 0 and a != 0):
            v_new += k * np.sign(a)
        v = float(np.clip(v_new, -vmax, vmax)); x += v * dt; pos[i] = x
    return np.interp(t, T, pos)

def score(f, d, g, k, horizon=None):
    t, arm, ct, ca = load(f)
    m = t <= (horizon or t[-1])
    p = simulate(t[m], ct, ca, d, g, k)
    return math.degrees(np.sqrt(np.mean((p - (arm[m] - arm[0])) ** 2)))

if __name__ == '__main__':
    train, held = 'runs/accel_reversal_2/capture.npz', ['runs/accel_reversal_1/capture.npz', 'runs/accel_step_full_1/capture.npz']
    print('simulator-like (d=18 ms, g=1, k=0):', round(score(train, .018, 1, 0), 2), 'deg on run 2')
    best = min(itertools.product((.006, .008, .010, .012, .014, .018), (0.9, 1.0, 1.1), (0, .1, .2, .3, .4, .5)),
               key=lambda p: score(train, *p))
    print('best on run 2:', best, round(score(train, *best), 2), 'deg')
    for h in held:
        print(f'  held-out {h.split("/")[1]}: model {score(h, *best):.2f} deg  vs simulator-like {score(h, .018, 1, 0):.2f} deg')
