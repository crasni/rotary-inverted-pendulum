"""Validate the zero-crossing kick through the real simulator path (sysid replay env),
fitted on reversal run 2, checked on reversal run 1 and the step capture."""
import sys, math, inspect, itertools
sys.path.insert(0, 'src')
import numpy as np
import pendulum_lab.sysid_accel as SA

def sim_arm(f, delay, gain, kick):
    d = np.load(f, allow_pickle=True); r = d['samples']; t0 = float(d['start_host_time_s'])
    t = r[:, 0] - t0; ct = d['command_t'] - (t0 if d['command_t'][0] > 1000 else 0)
    src = inspect.getsource(SA.replay).replace('transport_delay_s=.010', f'transport_delay_s={delay}')
    src = src.replace('        env.reset(seed=0)\n', f'        env.reset(seed=0)\n        env._accel_transport.zero_cross_kick_rad_s = {kick}\n')
    ns = dict(SA.__dict__); exec(src, ns)
    sim = ns['replay'](ct, np.asarray(d['accel_commands']) * gain, t, np.array([r[0, 3], r[0, 4], 0, 0]), 'configs/my_rig.json')
    return math.degrees(np.sqrt(np.mean((sim[:, 0] - r[:, 3]) ** 2)))

train = 'runs/accel_reversal_2/capture.npz'
held = ['runs/accel_reversal_1/capture.npz', 'runs/accel_step_full_1/capture.npz']
grid = list(itertools.product((0.010, 0.014, 0.018), (0.9, 1.0), (0.0, 0.2, 0.3, 0.4)))
scores = {g: sim_arm(train, *g) for g in grid}
best = min(scores, key=scores.get)
base = (0.018, 1.0, 0.0)
print(f'fit on reversal run 2: current model {scores[base]:.2f} deg, best {best} {scores[best]:.2f} deg')
for h in held:
    print(f'  held-out {h.split("/")[1]}: current {sim_arm(h, *base):.2f} deg  best {sim_arm(h, *best):.2f} deg')
