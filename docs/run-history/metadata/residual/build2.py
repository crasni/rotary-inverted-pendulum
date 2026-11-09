import sys, glob, math, json, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
import mujoco
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
DELAY=float(sys.argv[1]) if len(sys.argv)>1 else 0.010
runs=sorted(set(glob.glob('docs/results/hardware/*/2026*')+glob.glob('runs/sac_[ADFJ]_2026*')+glob.glob('runs/dqn_rig_2026*')+glob.glob('runs/ppo_rig_2026*')))
runs=[r for r in runs if '20261107_163412' not in r]
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':DELAY}); env.reset(seed=0); p=env.unwrapped
rows=[]; meta=[]
def tru(s,k):  # state with velocities from central differences of positions
    u=np.unwrap(s[k-1:k+2,1]); a=s[k-1:k+2,0]
    return s[k,0], s[k,1], (a[2]-a[0])/0.04, (u[2]-u[0])/0.04
for r in runs:
    try: s,a=R.load_run(r)
    except Exception: continue
    if len(s)<50: continue
    ri=len(meta); meta.append(r)
    for k in range(8,len(s)-3):
        R.set_state(env,s,a,k-3)
        for j in (k-3,k-2,k-1): env.step(np.array([a[j]],dtype=np.float32))
        arm,th,armv,penv=tru(s,k)
        p.data.qpos[p._pen_qpos_addr]=th+math.pi; p.data.qvel[p._pen_qvel_addr]=penv
        mujoco.mj_forward(p.model,p.data)
        armv_sim=float(p.data.qvel[p._motor_qvel_addr]); arm_sim=float(p.data.qpos[p._motor_qpos_addr])
        out=[]
        for j in (k,k+1):
            env.step(np.array([a[j]],dtype=np.float32)); out.append(math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi))
        rows.append([ri,k,arm,th,armv,penv,a[k],a[k-1],a[k-2],a[k-3],arm_sim,armv_sim,out[0],out[1],s[k+1,1],s[k+2,1],a[k+1]])
D=np.array(rows); np.savez(f'runs/residual/dataset2_{int(DELAY*1e3)}.npz', data=D, runs=np.array(meta)); print(D.shape,len(meta))
d1=np.array([math.remainder(x,2*math.pi) for x in D[:,14]-D[:,12]]); d2=np.array([math.remainder(x,2*math.pi) for x in D[:,15]-D[:,13]])
th=D[:,3]
for name,m in (('all',np.ones(len(th),bool)),('upright |th|<0.5',np.abs(th)<0.5),('hanging |th|>2.5',np.abs(th)>2.5)):
    print(f'{name:18s} theta error after 20 ms {np.degrees(np.sqrt(np.mean(d1[m]**2))):.2f} deg, after 40 ms {np.degrees(np.sqrt(np.mean(d2[m]**2))):.2f} deg')
