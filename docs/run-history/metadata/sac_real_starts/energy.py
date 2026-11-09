import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts','runs/sac_real_starts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import state_from_obs
from train_rig_rl import RIG
W2=11.30**2  # omega_n^2 from free swing fit
def crossings(th, w):
    """pendulum speed each time theta passes through upright (sign change near 0)"""
    out=[]
    for k in range(1,len(th)):
        if abs(th[k])<0.4 and abs(th[k-1])<0.4 and np.sign(th[k])!=np.sign(th[k-1]): out.append(abs(w[k]))
    return out
def first_cross(th,w):
    c=crossings(th,w); return c[0] if c else np.nan
for name in ('sac_A','sac_D'):
    fs=sorted(glob.glob(f'runs/{name}_*'))
    F=[];ALL=[]
    for f in fs:
        x=np.load(f+'/episode_000.npz')['samples']; c=crossings(x[:,2],x[:,4]); F.append(c[0] if c else np.nan); ALL+=c
    print(f'REAL {name}: first top crossing speed {np.round(F,1)} rad/s; all crossings median {np.median(ALL):.1f} (n={len(ALL)})')
for ckn,ck in (('A','runs/sac_real_starts/A/policy.zip'),('D','runs/sac_real_starts/D/policy.zip')):
    c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path(ck))
    for dl in (0.018,0.03,0.04):
        env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':dl}); p=env.unwrapped; p.exact_hanging_reset_frac=1.0
        F=[];ALL=[];T=[]
        for e in range(3):
            o,_=env.reset(seed=e); c.reset(); th=[];w=[]
            for k in range(750):
                o,*_=env.step(np.array([c.act(o)],dtype=np.float32)); x=state_from_obs(o); th.append(x[1]); w.append(x[3])
            cc=crossings(np.array(th),np.array(w)); F.append(cc[0] if cc else np.nan); ALL+=cc
        print(f'SIM {ckn} delay {dl*1e3:.0f}ms: first top crossing speed {np.round(F,1)}; total crossings in 3 eps {len(ALL)}')
