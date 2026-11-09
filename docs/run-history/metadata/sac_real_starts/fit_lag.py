import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from multiprocessing import Pool
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
RUNS=[r for r in sorted(glob.glob('docs/results/hardware/sac/2026*'))+sorted(glob.glob('runs/sac_[AD]_*')) if '163412' not in r]
def err(args):
    dl, lag = args
    env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':dl}); p=env.unwrapped; p.exact_hanging_reset_frac=1.0
    ep=[];ea=[]
    for r in RUNS:
        s,a=R.load_run(r); env.reset(seed=0); p._pen_obs_lag_s=lag; p.seed_measurement_ring()
        th=[];av=[]
        for k in range(60):  # 1.2 s, before the runs diverge chaotically
            o,*_=env.step(np.array([a[k]],dtype=np.float32)); th.append(math.atan2(o[-5],o[-4])); av.append(o[-3])
        rt=s[1:61,1]; ep.append(np.sqrt(np.mean(np.array([math.remainder(x-y,2*math.pi) for x,y in zip(th,rt)])**2))); ea.append(np.sqrt(np.mean((np.array(av)-s[1:61,2])**2)))
    return dl, lag, math.degrees(np.mean(ep)), np.mean(ea)
if __name__=='__main__':
    grid=[(d,l) for d in (0.004,0.008,0.012,0.018,0.024,0.030) for l in (0.0,0.01,0.02,0.03,0.04)]
    with Pool(14) as pool: res=pool.map(err,grid)
    res.sort(key=lambda r:r[2]+10*r[3])
    for dl,lag,e,a in res[:10]: print('motor delay %2d ms, pendulum reading lag %2d ms: measured pendulum err %.2f deg, arm vel err %.2f rad/s'%(dl*1e3,lag*1e3,e,a))
