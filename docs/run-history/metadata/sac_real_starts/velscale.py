import sys, glob, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import state_from_obs
from train_rig_rl import RIG
def fit(th, w):
    u=np.unwrap(th); cd=(u[2:]-u[:-2])/0.04; v=w[1:-1]
    m=np.abs(cd)>3
    slope=np.sum(v[m]*cd[m])/np.sum(cd[m]**2)
    # one-sided: backward difference (what the reading should match if it lags 10 ms)
    bd=(u[1:-1]-u[:-2])/0.02
    sb=np.sum(v[m]*bd[m])/np.sum(bd[m]**2)
    return slope, sb, m.sum()
R=[]
for f in [r for r in sorted(glob.glob('docs/results/hardware/sac/2026*'))+sorted(glob.glob('runs/sac_[AD]_*')) if '163412' not in r]:
    x=np.load((f+'/episode_000.npz'))['samples']; R.append(fit(x[:,2],x[:,4]))
R=np.array(R); print('REAL: reading / central-diff %.3f, reading / backward-diff %.3f (per run %s)'%(R[:,0].mean(),R[:,1].mean(),np.round(R[:,0],3)))
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('runs/sac_real_starts/D/policy.zip'))
env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.03}); p=env.unwrapped; p.exact_hanging_reset_frac=0.0
S=[]
for e in range(4):
    o,_=env.reset(seed=e); c.reset(); th=[];w=[]
    for k in range(400):
        o,*_=env.step(np.array([c.act(o)],dtype=np.float32)); x=state_from_obs(o); th.append(x[1]); w.append(x[3])
    S.append(fit(np.array(th),np.array(w)))
S=np.array(S); print('SIM : reading / central-diff %.3f, reading / backward-diff %.3f'%(S[:,0].mean(),S[:,1].mean()))
