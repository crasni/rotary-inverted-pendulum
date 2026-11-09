import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import state_from_obs
from train_rig_rl import RIG
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('runs/sac_real_starts/D/policy.zip'))
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.03}); p=env.unwrapped; p.exact_hanging_reset_frac=1.0
o,_=env.reset(seed=0); c.reset(); sim=[]
for k in range(130):
    x=state_from_obs(o); u=c.act(o); sim.append([*x,u]); o,*_=env.step(np.array([u],dtype=np.float32))
sim=np.array(sim)
for f in sorted(glob.glob('runs/sac_D_*'))[:2]:
    r=np.load(f+'/episode_000.npz')['samples'][:130,1:6]
    print(f[-15:])
    idx=list(range(0,130,8))
    print(' t        ', np.array(idx)*0.02)
    print(' cmd real ', r[idx,4]); print(' cmd sim  ', sim[idx,4])
    print(' pend real', np.degrees(np.unwrap(r[:,1]+math.pi))[idx]-0); print(' pend sim ', np.degrees(np.unwrap(sim[:,1]+math.pi))[idx])
    print(' arm real ', r[idx,0]); print(' arm sim  ', sim[idx,0])
