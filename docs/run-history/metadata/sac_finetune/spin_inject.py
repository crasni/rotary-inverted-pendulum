import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from train_rig_rl import RIG
import real_state_starts as R
W2=11.30**2; TOP=2*W2
E=lambda th,w:(0.5*w*w+W2*(1-math.cos(th+math.pi)))/TOP
for name,ck in (('R50','runs/sac_finetune/R50/policy.zip'),('current','docs/results/rig_rl_disturbance/sac_long/policy.zip')):
    c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path(ck))
    env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env.reset(seed=0); p=env.unwrapped
    caught=0; n=0; Eend=[]
    for d in sorted(glob.glob('runs/sac_R50_*')):
        s,a=R.load_run(d)
        for k in range(150,len(s)-10,25):
            if E(s[k,1],s[k,3])<1.0: continue
            o=R.set_state(env,s,a,k); c.reset(); th=[]; Es=[]
            for j in range(200):
                o,_,te,_,_=env.step(np.array([c.act(o)],dtype=np.float32))
                t=math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi); th.append(t); Es.append(E(t,float(p.data.qvel[p._pen_qvel_addr])))
                if te: break
            n+=1; th=np.array(th); caught+= len(th)==200 and bool(np.all(np.abs(th[-100:])<math.radians(15))); Eend.append(np.mean(Es[50:100]))
    print(f'{name}: from {n} real mid-spin states (energy>1.0x), sim catches within 4 s: {caught}/{n}; sim energy 1-2 s later mean {np.mean(Eend):.2f}x top (rig: stays ~1.15x)')
