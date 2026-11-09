import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
RUNS=[r for r in sorted(glob.glob('docs/results/hardware/sac/2026*'))+sorted(glob.glob('runs/sac_[AD]_*')) if '163412' not in r]
for ma in (50,55,60,65):
  for dl in (0.008,0.012):
    env=stress_env(Path("configs/my_rig.json"),{**RIG,"delay_s":dl,"max_accel":ma,"speed_limit":4.9}); p=env.unwrapped; p.exact_hanging_reset_frac=1.0; p.max_accel_rad_s2=float(ma)
    ea=[];ep=[];eth=[]
    for r in RUNS:
        s,a=R.load_run(r); env.reset(seed=0)
        av=[];ap=[];th=[]
        for k in range(60):
            # policy commands are normalized by 50 rad/s^2 on the rig; keep that meaning
            o,*_=env.step(np.array([np.clip(a[k]*50/ma,-1,1) if False else a[k]],dtype=np.float32)); av.append(o[-3]); ap.append(o[-6]); th.append(math.atan2(o[-5],o[-4]))
        ea.append(np.sqrt(np.mean((np.array(av)-s[1:61,2])**2))); ep.append(np.sqrt(np.mean((np.array(ap)-s[1:61,0])**2)))
        eth.append(np.sqrt(np.mean(np.array([math.remainder(x-y,2*math.pi) for x,y in zip(th,s[1:61,1])])**2)))
    print('arm max accel %d (cmd 1.0 -> %d rad/s^2), motor delay %2d ms: arm vel err %.2f, arm pos err %.3f rad, pendulum err %.1f deg'%(ma,ma,dl*1e3,np.mean(ea),np.mean(ep),math.degrees(np.mean(eth))))
