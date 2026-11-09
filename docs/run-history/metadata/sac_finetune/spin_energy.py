import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
import mujoco
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
W2=11.30**2; TOP=2*W2
E=lambda th,w:(0.5*w*w+W2*(1-math.cos(th+math.pi)))/TOP
def tru(s,k):
    u=np.unwrap(s[k-1:k+2,1]); a=s[k-1:k+2,0]; return s[k,0],s[k,1],(a[2]-a[0])/0.04,(u[2]-u[0])/0.04
for delay in (0.010,0.018,0.030):
    env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':delay}); env.reset(seed=0); p=env.unwrapped
    dr=[];ds=[]
    for d in sorted(glob.glob('runs/sac_R50_*'))+sorted(glob.glob('runs/sac_[ADF]_*'))+sorted(glob.glob('docs/results/hardware/sac/2026*')):
        if '163412' in d: continue
        s,a=R.load_run(d)
        for k in range(150,len(s)-20,7):
            arm,th,armv,penv=tru(s,k)
            if E(th,penv)<1.0 or abs(penv)<6: continue   # spinning only
            R.set_state(env,s,a,k-3)
            for j in (k-3,k-2,k-1): env.step(np.array([a[j]],dtype=np.float32))
            p.data.qpos[p._pen_qpos_addr]=th+math.pi; p.data.qvel[p._pen_qvel_addr]=penv; mujoco.mj_forward(p.model,p.data)
            for j in range(15): env.step(np.array([a[k+j]],dtype=np.float32))
            ths=float(p.data.qpos[p._pen_qpos_addr])-math.pi; ws=float(p.data.qvel[p._pen_qvel_addr])
            _,thr,_,wr=tru(s,k+15)
            e0=E(th,penv); dr.append(E(thr,wr)-e0); ds.append(E(ths,ws)-e0)
    dr=np.array(dr); ds=np.array(ds)
    slope=np.sum(dr*ds)/np.sum(ds*ds)
    print(f'delay {delay*1e3:.0f} ms: {len(dr)} spinning segments, energy change over 0.3 s  real mean {dr.mean():+.3f}  sim mean {ds.mean():+.3f} (x top) | corr {np.corrcoef(dr,ds)[0,1]:.2f}, real = {slope:.2f} x sim | when sim removes energy (<-0.05): real mean {dr[ds<-0.05].mean():+.3f} vs sim {ds[ds<-0.05].mean():+.3f}')
