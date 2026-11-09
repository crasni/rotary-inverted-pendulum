import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import observation_frame
from train_rig_rl import RIG
import real_state_starts as R
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('runs/sac_real_starts/F/policy.zip'))
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env.reset(seed=0); p=env.unwrapped
sim_j=[]; real_j=[]; first_step_same=[]
for d in sorted(glob.glob('runs/sac_F_*')):
    s,a=R.load_run(d)
    for k in [k for k in R.near_upright(s,theta_max=0.3,last=25) if abs(s[k,3])<3]:
        p._pen_obs_lag_s=0.010; o=R.set_state(env,s,a,k); c.reset(); acts=[]
        for j in range(10):
            u=c.act(o); acts.append(u); o,*_=env.step(np.array([u],dtype=np.float32))
        acts=np.array(acts); ra=a[k:k+10]
        sim_j.append(np.mean(np.abs(np.diff(acts))>0.5)); real_j.append(np.mean(np.abs(np.diff(ra))>0.5))
        first_step_same.append(abs(acts[0]-ra[0]))
print('F from the same real upright states (n=%d): big jumps next 0.2 s  sim %.0f%%  vs rig %.0f%%; first action sim vs rig mean diff %.2f'%(len(sim_j),100*np.mean(sim_j),100*np.mean(real_j),np.mean(first_step_same)))
# Which input drives the jumps on the rig? replay real obs, perturb one channel at a time, see action change
names=['arm','sin','cos','arm_vel','pen_vel','prev_action']
X=[]
for d in sorted(glob.glob('runs/sac_F_*')):
    s,a=R.load_run(d)
    for k in R.near_upright(s,theta_max=0.3):
        X.append(R.history(s,a,k))
X=np.array(X); base=np.array([c.controller.act(x) if hasattr(c,'controller') else c.act(x) for x in X])
for i,n in enumerate(names):
    for delta in ({'arm':0.05,'sin':0.02,'cos':0.0,'arm_vel':0.3,'pen_vel':0.3,'prev_action':0.2}[n],):
        if delta==0: continue
        Y=X.copy(); Y[:,i::6]+=delta
        if n=='sin': Y[:,2::6]=np.sqrt(np.clip(1-Y[:,1::6]**2,0,1))*np.sign(X[:,2::6])
        new=np.array([c.controller.act(y) for y in Y]); print(f'  +{delta} on {n:11s} (all 4 frames): mean |action change| {np.mean(np.abs(new-base)):.2f}')
