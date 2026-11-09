import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env.reset(seed=0); p=env.unwrapped
rows=[]
for name,pat in (('LQR','docs/results/hardware/lqr/2026*'),('SAC','docs/results/hardware/sac/2026*'),('SAC','runs/sac_[AD]_*')):
    for d in sorted(glob.glob(pat)):
        try: s,a=R.load_run(d)
        except Exception: continue
        for k in range(5,len(s)-4,2):
            if abs(s[k,1])>0.5: continue
            R.set_state(env,s,a,k)
            for j in range(3): o,*_=env.step(np.array([a[k+j]],dtype=np.float32))
            e=o[-3]-s[k+3,2]   # arm velocity after 3 steps (60 ms), sim minus real
            jump=max(abs(a[k+j]-a[k+j-1]) for j in range(3))
            rows.append((name,jump,e))
import collections
for name in ('LQR','SAC'):
    r=np.array([(j,e) for n,j,e in rows if n==name])
    bins=[0,0.2,0.5,1.0,2.01]
    out=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(r[:,0]>=lo)&(r[:,0]<hi)
        out.append(f'|Δcmd| {lo}-{hi}: {np.sqrt(np.mean(r[m,1]**2)):.2f} rad/s (n={m.sum()})' if m.sum() else f'{lo}-{hi}: -')
    print(f'{name}: share of steps with |Δcmd|>0.5: {np.mean(r[:,0]>0.5):.0%}')
    print('   sim arm-velocity error after 60 ms, by largest command jump:', ' | '.join(out))
