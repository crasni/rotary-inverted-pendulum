import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from train_rig_rl import RIG
import real_state_starts as R
W2=11.30**2
def E(th,w):  # energy per inertia, 0 hanging at rest, 2*W2 at upright at rest
    phi=th+math.pi; return 0.5*w*w + W2*(1-math.cos(phi))
TOP=2*W2
def first_arrival(s):
    for k in range(1,len(s)):
        if abs(s[k,1])<0.4 and np.sign(s[k,1])!=np.sign(s[k-1,1]): return k
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('runs/sac_finetune/R50/policy.zip'))
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env.reset(seed=0); p=env.unwrapped
for d in sorted(glob.glob('runs/sac_R50_*'))+sorted(glob.glob('runs/sac_D_*'))[:2]:
    s,a=R.load_run(d); ka=first_arrival(s)
    real_E=E(s[ka,1],s[ka,3])
    print(f'{d[5:]}: first arrival t={ka*0.02:.2f}s, real energy at arrival {real_E/TOP:.2f}x top (speed {abs(s[ka,3]):.1f})')
    for back in (5,10,15,25,40):
        k=ka-back
        if k<8: continue
        o=R.set_state(env,s,a,k); c.reset(); Es=[]; acts=[]
        e0=E(s[k,1],s[k,3])
        for j in range(back+10):
            u=c.act(o); acts.append(u); o,*_=env.step(np.array([u],dtype=np.float32))
            Es.append(E(float(p.data.qpos[p._pen_qpos_addr])-math.pi, float(p.data.qvel[p._pen_qvel_addr])))
        # sim energy at the same time as the real arrival
        print(f'   start {back*0.02:.2f}s before (real energy then {e0/TOP:.2f}x): sim energy at arrival time {Es[back-1]/TOP:.2f}x | real {real_E/TOP:.2f}x | real cmds vs sim cmds mean |diff| {np.mean(np.abs(np.array(acts[:back])-a[k:k+back])):.2f}')
