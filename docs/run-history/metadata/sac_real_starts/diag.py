import sys, math, numpy as np
sys.path[:0]=["src","scripts"]
from pathlib import Path
import gymnasium as gym
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
# 1) does the training env actually start from the bank?
env=stress_env(Path('configs/my_rig.json'),RIG); p=env.unwrapped
p._spin_reset_obs=np.load('runs/sac_real_starts/bank_train.npz')['obs']; p.spin_reset_frac=0.4
p.exact_hanging_reset_frac=0.18; p.upright_reset_frac=0.12
th=[]; vel=[]
for i in range(500):
    o,_=env.reset(seed=i); th.append(math.atan2(o[-5],o[-4])); vel.append(o[-2])
th=np.abs(th); vel=np.abs(vel)
print('resets: near upright %.0f%%, of which |pen vel|>2: %.0f%%'%(100*np.mean(th<1),100*np.mean(vel[th<1]>2)))
# 2) held-out states by pendulum speed
import real_state_starts as R
for f in ['docs/results/hardware/sac/20261107_163427','docs/results/hardware/sac/20261107_165745']:
    s,a=R.load_run(f); ks=R.near_upright(s,theta_max=0.35,last=100)
    v=np.abs(s[ks,3]); print(f[-6:],'test states',len(ks),'|pen vel| <2: %d, 2-4: %d, >4: %d'%((v<2).sum(),((v>=2)&(v<4)).sum(),(v>=4).sum()))
