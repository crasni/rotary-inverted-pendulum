import sys, glob, math, json, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
import mujoco
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
runs=sorted(set(glob.glob('docs/results/hardware/*/2026*')+glob.glob('runs/sac_[ADF]_2026*')+glob.glob('runs/dqn_rig_2026*')+glob.glob('runs/ppo_rig_2026*')))
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env.reset(seed=0); p=env.unwrapped
rows=[]; meta=[]
for ri,r in enumerate(runs):
    try:
        s,a=R.load_run(r)
        if json.load(open(r+'/summary.json'))[0]['stop_reason'].startswith('Keyboard') and 'sac/20261107_163412' in r: continue
    except Exception: continue
    if len(s)<50: continue
    for k in range(8,len(s)-1):
        R.set_state(env,s,a,k-3)
        for j in (k-3,k-2,k-1):
            env.step(np.array([a[j]],dtype=np.float32))
        # overwrite the pendulum with the real state at k; keep arm/planner from the replay
        p.data.qpos[p._pen_qpos_addr]=s[k,1]+math.pi; p.data.qvel[p._pen_qvel_addr]=s[k,3]
        mujoco.mj_forward(p.model,p.data); 
        arm_sim_k=float(p.data.qpos[p._motor_qpos_addr]); armv_sim_k=float(p.data.qvel[p._motor_qvel_addr])
        o,*_=env.step(np.array([a[k]],dtype=np.float32))
        th_sim=math.atan2(o[-5],o[-4]); w_sim=float(p.data.qvel[p._pen_qvel_addr]); th_true=math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi)
        rows.append([ri, k, *s[k], a[k], a[k-1], a[k-2], a[k-3], arm_sim_k, armv_sim_k, th_true, w_sim, *s[k+1]])
    meta.append(r)
D=np.array(rows)
np.savez('runs/residual/dataset.npz', data=D, runs=np.array(meta))
print(D.shape, 'runs', len(meta))
