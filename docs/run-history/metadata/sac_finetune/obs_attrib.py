import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from train_rig_rl import RIG
import real_state_starts as R
W2=11.30**2; TOP=2*W2
E=lambda th,w:(0.5*w*w+W2*(1-math.cos(th+math.pi)))/TOP
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('runs/sac_finetune/R50/policy.zip')); pol=c.controller
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env.reset(seed=0); p=env.unwrapped
names=['arm','sin','cos','arm_vel','pen_vel']
for regime,cond in (('spinning',lambda th,w:E(th,w)>1.0 and abs(w)>6),('near upright slow',lambda th,w:abs(th)<0.4 and abs(w)<3)):
    diffs=[]; chan=np.zeros((5,)); total=[]; ch_err=[]
    for d in sorted(glob.glob('runs/sac_R50_*'))+sorted(glob.glob('runs/sac_[ADF]_*')):
        s,a=R.load_run(d)
        for k in range(10,len(s)-3,3):
            if not cond(s[k,1],s[k,3]): continue
            o=R.set_state(env,s,a,k)
            for j in range(2): o,*_=env.step(np.array([a[k+j]],dtype=np.float32))
            real=R.history(s,a,k+2)
            sim=np.asarray(o,dtype=np.float32)
            ar=pol.act(real); as_=pol.act(sim); total.append(abs(ar-as_))
            fr=real[-6:]; fs=sim[-6:]; ch_err.append(np.abs(fr-fs)[:5])
            for i,n in enumerate(names):
                x=sim.copy(); x[i::6]=real[i::6]  # put the real channel into the sim observation (all 4 frames)
                chan[i]+=abs(pol.act(x)-as_)
    m=len(total)
    print(f'{regime} ({m} states): |action(real obs) - action(sim obs)| after 2 steps = {np.mean(total):.2f}')
    print('   mean |real-sim| per channel (newest frame):', dict(zip(names,np.round(np.mean(ch_err,0),3))))
    print('   action change from swapping in the real channel:', dict(zip(names,np.round(chan/m,2))))
