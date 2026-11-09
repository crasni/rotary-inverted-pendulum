import sys, numpy as np
sys.path[:0]=['src','scripts','runs/sac_real_starts']
from pathlib import Path
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import state_from_obs
from train_rig_rl import RIG, RandomPhysics
from energy import crossings
ck=sys.argv[1]
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path(ck))
base=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.03}); env=RandomPhysics(base); base.unwrapped.exact_hanging_reset_frac=1.0
F=[];ok=0
for e in range(12):
    o,info=env.reset(seed=900+e); c.reset(); th=[];w=[]
    for k in range(750):
        o,*_=env.step(np.array([c.act(o)],dtype=np.float32)); x=state_from_obs(o); th.append(x[1]); w.append(x[3])
    th=np.array(th); cc=crossings(th,np.array(w)); F.append(cc[0] if cc else np.nan)
    ok+=bool(np.all(np.abs(th[-500:])<np.radians(15)))
print(ck, 'first top-crossing speeds', np.round(F,1), '| held final 10 s', ok,'/12')
