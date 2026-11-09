import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import wrap
from train_rig_rl import RIG
import real_state_starts as R
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('runs/sac_real_starts/A/policy.zip'))
for dl in (0.018,0.030,0.040):
    env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':dl}); env.reset(seed=0); p=env.unwrapped
    simsat=[];realsat=[];simsw=[];realsw=[]
    for r in sorted(glob.glob('runs/sac_A_*')):
        s,a=R.load_run(r)
        for k in [k for k in R.near_upright(s,theta_max=0.2,last=25) if abs(s[k,3])<2][::2]:
            o=R.set_state(env,s,a,k); c.reset(); acts=[]
            for j in range(20):
                u=c.act(o); acts.append(u); o,*_=env.step(np.array([u],dtype=np.float32))
            acts=np.array(acts); ra=a[k:k+20]
            simsat.append(np.mean(np.abs(acts)>0.95)); realsat.append(np.mean(np.abs(ra)>0.95))
            simsw.append(np.mean(np.abs(np.diff(acts))>1.0)); realsw.append(np.mean(np.abs(np.diff(ra))>1.0))
    print('delay %d ms, %d calm real upright states: next 0.4 s full strength sim %.0f%% vs real %.0f%% | big swings (|Δa|>1) sim %.0f%% vs real %.0f%%'%(dl*1e3,len(simsat),100*np.mean(simsat),100*np.mean(realsat),100*np.mean(simsw),100*np.mean(realsw)))
