import sys, math, inspect
sys.path[:0] = ['src']
import numpy as np
import pendulum_lab.sysid_accel as SA
f = sys.argv[1]
d = np.load(f, allow_pickle=True); r = d['samples']; t0 = float(d['start_host_time_s'])
t = r[:, 0] - t0; ct = d['command_t']; ca = d['accel_commands']
if ct[0] > 1000: ct = ct - t0
print('commands:', ' '.join(f'{x:.3f}:{a:+.0f}' for x, a in zip(ct, ca) if a != 0))
initial = np.array([r[0, 3], r[0, 4], 0.0, 0.0])
for delay in (0.010, 0.018):
    src = inspect.getsource(SA.replay).replace('transport_delay_s=.010', f'transport_delay_s={delay}')
    ns = dict(SA.__dict__); exec(src, ns)
    sim = ns['replay'](ct, ca, t, initial, 'configs/my_rig.json')
    print(f'--- delay {delay*1e3:.0f} ms: time, arm real/sim (deg), pendulum real/sim (deg)')
    for k in range(0, len(t), 10):
        print(f'  t={t[k]:5.2f} arm {math.degrees(r[k,3]):+6.1f}/{math.degrees(sim[k,0]):+6.1f}  pen {math.degrees(math.remainder(r[k,4],2*math.pi)):+6.2f}/{math.degrees(math.remainder(sim[k,1],2*math.pi)):+6.2f}')

# --- correct for the pendulum's pre-existing free swing (linear superposition, small angles)
src = inspect.getsource(SA.replay).replace('transport_delay_s=.010', 'transport_delay_s=0.018')
ns = dict(SA.__dict__); exec(src, ns)
sim = ns['replay'](ct, ca, t, initial, 'configs/my_rig.json')
real_p = np.unwrap(r[:, 4]); sim_p = np.unwrap(sim[:, 1])
w = 12.27; zeta = 0.01
pre = t < ct[np.argmax(ca != 0)] + 0.01
basis = lambda tt: np.c_[np.exp(-zeta*w*tt)*np.cos(w*tt), np.exp(-zeta*w*tt)*np.sin(w*tt)]
coef, *_ = np.linalg.lstsq(basis(t[pre]), (real_p - sim_p)[pre], rcond=None)
free = basis(t) @ coef
forced_real = real_p - free
print('pre-swing amplitude (deg):', np.degrees(np.hypot(*coef)).round(2), ' pre-pulse fit residual (deg):', np.degrees(np.std((real_p-sim_p-free)[pre])).round(2))
for lo, hi, name in ((0.3, 1.0, 'pulse 1 (+-16.7 x5 ticks)'), (1.0, 1.6, 'pulse 2 (+33 x3, -33 x2)'), (1.6, t[-1], 'pulse 3 (+50 x2, -50 x1)')):
    m = (t >= lo) & (t < hi)
    a_r = forced_real[m] - forced_real[m][0]; a_s = sim_p[m] - sim_p[m][0]
    g = float(a_r @ a_s / (a_s @ a_s))
    lags = np.arange(-8, 9); best = min(lags, key=lambda L: np.mean((np.roll(a_s, L) - a_r)[10:-10]**2))
    print(f'{name:28s}: pendulum response real/sim gain {g:.2f}; rms diff {np.degrees(np.sqrt(np.mean((a_r-a_s)**2))):.2f}° (sim swing rms {np.degrees(np.std(a_s)):.2f}°); best shift {best*5:+d} ms')
    print(f'   arm travel real {np.degrees(r[m,3][-1]-r[m,3][0]):+.1f}° sim {np.degrees(sim[m,0][-1]-sim[m,0][0]):+.1f}°')
