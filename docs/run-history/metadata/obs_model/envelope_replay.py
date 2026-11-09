"""Replay the 50 rad/s^2 driven sweeps (pendulum hanging, from rest) in the simulator.
No initial-state estimation: the rig started still. Compare arm and pendulum."""
import sys, glob, math
sys.path[:0] = ['src']
import numpy as np
from pendulum_lab.sysid_accel import replay as sysid_replay
import pendulum_lab.sysid_accel as SA
from pendulum_lab.plant.pendulum_env import RotaryInvertedPendulumEnv
import inspect

def run(f, delay):
    d = np.load(f, allow_pickle=True); rows = d['samples']
    t = rows[:, 0] - float(d['start_host_time_s']); keep = t >= 0; rows, t = rows[keep], t[keep]
    initial = np.array([rows[0, 3], rows[0, 4], 0.0, 0.0])
    src = inspect.getsource(SA.replay).replace('transport_delay_s=.010', f'transport_delay_s={delay}')
    ns = dict(SA.__dict__); exec(src, ns)
    sim = ns['replay'](d['command_t'], d['accel_commands'], t, initial, 'configs/my_rig.json')
    em = rows[:, 3] - sim[:, 0]; ep = np.vectorize(math.remainder)(rows[:, 4] - sim[:, 1], 2 * math.pi)
    amp = np.degrees(np.max(np.abs(np.vectorize(math.remainder)(rows[:, 4] - math.pi, 2*math.pi) ))) if False else np.degrees(np.ptp(np.unwrap(rows[:, 4])))
    return np.degrees(np.sqrt(np.mean(em ** 2))), np.degrees(np.sqrt(np.mean(ep ** 2))), amp, t[-1]

for f in sorted(glob.glob('docs/results/sysid/driven/envelope*/*.npz')) + sorted(glob.glob('docs/results/sysid/sysid_motor_envelope*/*.npz')):
    out = [run(f, dl) for dl in (0.010, 0.018, 0.026)]
    print(f'{f[26:]:40s} pendulum swing p2p {out[0][2]:5.1f}° over {out[0][3]:.1f}s | arm/pendulum RMSE (deg) delay10: {out[0][0]:.2f}/{out[0][1]:.2f}  delay18: {out[1][0]:.2f}/{out[1][1]:.2f}  delay26: {out[2][0]:.2f}/{out[2][1]:.2f}')
