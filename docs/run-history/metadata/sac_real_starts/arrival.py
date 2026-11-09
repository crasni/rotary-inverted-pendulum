import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
import mujoco
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import wrap, state_from_obs
from train_rig_rl import RIG
import real_state_starts as R
env=stress_env(Path('configs/my_rig.json'),RIG); p=env.unwrapped
def th(): return wrap(float(p.data.qpos[p._pen_qpos_addr])-math.pi)
def first_arrival(T, V, A):
    for i in range(len(T)):
        if abs(T[i])<0.35: return V[i], A[i], i*0.02
    return None
runs=sorted(glob.glob('docs/results/hardware/sac/2026*'))
runs=[r for r in runs if '163412' not in r]
print('REAL first arrival at upright vs SIM replaying the SAME commands from hanging (open loop):')
for r in runs:
    s,a=R.load_run(r)
    p.exact_hanging_reset_frac=1.0; env.reset(seed=0)
    T,V,A=[],[],[]
    for k in range(200):
        env.step(np.array([a[k]],dtype=np.float32)); T.append(th()); V.append(float(p.data.qvel[p._pen_qvel_addr])); A.append(float(p.data.qpos[p._motor_qpos_addr]))
    real=first_arrival(s[1:,1],s[1:,3],s[1:,0]); sim=first_arrival(T,V,A)
    # peak swing amplitude before arrival
    print(f"  {r[-15:]}: real t={real[2]:.2f}s penvel={real[0]:+.1f} arm={real[1]:+.2f} | sim t={sim[2] if sim else float('nan'):.2f}s penvel={(sim[0] if sim else float('nan')):+.1f}")
    # energy-like: max |theta from hanging| per half swing in first 2 s
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('docs/results/rig_rl_disturbance/sac_long/policy.zip'))
print('SIM closed-loop SAC first arrival:')
for seed in range(5):
    p.exact_hanging_reset_frac=1.0 if seed==0 else 0.0
    o,_=env.reset(seed=seed); c.reset(); T,V,A=[],[],[]
    for k in range(250):
        o,*_=env.step(np.array([c.act(o)],dtype=np.float32)); T.append(th()); V.append(float(p.data.qvel[p._pen_qvel_addr])); A.append(float(p.data.qpos[p._motor_qpos_addr]))
    f=first_arrival(T,V,A); print(f"  seed {seed}: t={f[2]:.2f}s penvel={f[0]:+.1f} arm={f[1]:+.2f}")
