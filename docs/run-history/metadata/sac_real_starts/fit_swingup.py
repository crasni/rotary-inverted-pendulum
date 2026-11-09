import sys, glob, math, itertools, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
import mujoco
from pendulum_lab.plant import pendulum_env as PE
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
_step = mujoco.mj_step
STATE = {"c": 1.0, "plant": None, "M": None}
def hooked(m, d):
    p = STATE["plant"]
    if p is not None and STATE["c"] != 1.0 and m is p.model:
        M = STATE["M"]; mujoco.mj_fullM(m, d, M)
        a, q = p._motor_qvel_addr, p._pen_qvel_addr
        d.qfrc_applied[q] = (1 - STATE["c"]) * M[q, a] * d.qacc[a]
    _step(m, d)
mujoco.mj_step = hooked
env = stress_env(Path('configs/my_rig.json'), RIG); p = env.unwrapped
STATE["plant"] = p; STATE["M"] = np.zeros((p.model.nv, p.model.nv))
base_damp = p.params.pendulum_friction
runs = [r for r in sorted(glob.glob('docs/results/hardware/sac/2026*')) if '163412' not in r]
data = [R.load_run(r) for r in runs]
N = 90  # 1.8 s, before any arrival at upright
def err(c, dmul, delay):
    STATE["c"] = c; p._fixed_transport_delay_s = delay; p.exact_hanging_reset_frac = 1.0
    out = []
    for s, a in data:
        env.reset(seed=0); p.model.dof_damping[p._pen_dof_addr] = base_damp * dmul
        th = []
        for k in range(N):
            env.step(np.array([a[k]], dtype=np.float32))
            th.append(float(p.data.qpos[p._pen_qpos_addr]))  # 0 = hanging
        real = np.unwrap(s[1:N+1, 1] + math.pi); real -= 2*math.pi*np.round(real[0]/(2*math.pi))
        out.append(np.sqrt(np.mean((np.array(th) - real)**2)))
    return float(np.degrees(np.mean(out)))
print('nominal sim (c=1, damping x1, 18 ms): %.1f deg rms' % err(1, 1, 0.018))
res = []
for c, dm, dl in itertools.product((0.75, 0.8, 0.85, 0.9, 0.95, 1.0), (1, 2, 3, 4.5, 6), (0.010, 0.018, 0.030)):
    res.append((err(c, dm, dl), c, dm, dl))
res.sort()
for r in res[:8]: print('rms %.1f deg  coupling %.2f  damping x%.1f  delay %.0f ms' % (r[0], r[1], r[2], r[3]*1e3))
