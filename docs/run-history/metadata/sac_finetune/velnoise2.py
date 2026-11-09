import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import state_from_obs
from train_rig_rl import RIG
def resid(th,w):
    u=np.unwrap(th); cd=(u[2:]-u[:-2])/0.04; return w[1:-1]-cd, cd
bins=[(0,0.5),(0.5,3),(3,8),(8,30)]
def table(R,S):
    return ' | '.join(f'{lo}-{hi} rad/s: {np.sqrt(np.mean(R[(np.abs(S)>=lo)&(np.abs(S)<hi)]**2)):.2f}' for lo,hi in bins if ((np.abs(S)>=lo)&(np.abs(S)<hi)).sum()>20)
for name,pat in (('RIG SAC runs','runs/sac_[ADFJ]_*'),('RIG R50','runs/sac_R50_*'),('RIG LQR','docs/results/hardware/lqr/2026*')):
    RR=[];SS=[]
    for d in sorted(glob.glob(pat)):
        try: x=np.load(d+'/episode_000.npz')['samples']
        except Exception: continue
        r,s=resid(x[:,2],x[:,4]); RR+=list(r); SS+=list(s)
    print(f'{name:12s} velocity reading minus angle-derived velocity, rms by speed:', table(np.array(RR),np.array(SS)))
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('runs/sac_finetune/R50/policy.zip'))
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); p=env.unwrapped; p.exact_hanging_reset_frac=0.0
RR=[];SS=[]
for e in range(4):
    o,_=env.reset(seed=e); c.reset(); th=[];w=[]
    p.recovery_reset_frac=0.5
    for k in range(750):
        x=state_from_obs(o); th.append(x[1]); w.append(x[3]); o,*_=env.step(np.array([c.act(o)],dtype=np.float32))
    r,s=resid(np.array(th),np.array(w)); RR+=list(r); SS+=list(s)
print(f'{"SIM R50":12s} velocity reading minus angle-derived velocity, rms by speed:', table(np.array(RR),np.array(SS)))
# at rest: first 3 samples of every run (pendulum hanging still)
rest=[]
for d in glob.glob('docs/results/hardware/*/2026*')+glob.glob('runs/sac_*_2026*'):
    try: x=np.load(d+'/episode_000.npz')['samples']; rest+=list(x[:3,4])
    except Exception: pass
print('RIG at rest (first samples): velocity reading rms %.3f rad/s, values seen %s'%(np.sqrt(np.mean(np.square(rest))),np.unique(np.round(rest,2))[:12]))
