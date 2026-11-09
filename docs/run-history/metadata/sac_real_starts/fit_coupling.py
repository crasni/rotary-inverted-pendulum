import sys, glob, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
from multiprocessing import Pool
from pendulum_lab.robustness import stress_env
from train_rig_rl import RIG
import real_state_starts as R
RUNS=[r for r in sorted(glob.glob('docs/results/hardware/sac/2026*'))+sorted(glob.glob('runs/sac_[AD]_*')) if '163412' not in r]
def err(args):
    c, dl, lag = args
    env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':dl}); p=env.unwrapped; p.exact_hanging_reset_frac=1.0
    ep=[];ea=[]
    for r in RUNS:
        s,a=R.load_run(r); env.reset(seed=0); p._coupling_gain=c; p._pen_obs_lag_s=lag; p.seed_measurement_ring()
        th=[];av=[]
        for k in range(75):
            o,*_=env.step(np.array([a[k]],dtype=np.float32)); th.append(math.atan2(o[-5],o[-4])); av.append(o[-3])
        ep.append(np.sqrt(np.mean(np.array([math.remainder(x-y,2*math.pi) for x,y in zip(th,s[1:76,1])])**2))); ea.append(np.sqrt(np.mean((np.array(av)-s[1:76,2])**2)))
    return c, dl, lag, math.degrees(np.mean(ep)), np.mean(ea)
if __name__=='__main__':
    grid=[(c,d,l) for c in (0.75,0.8,0.85,0.9,0.95,1.0) for d in (0.008,0.012,0.018) for l in (0.0,0.01,0.02)]
    with Pool(14) as pool: res=pool.map(err,grid)
    res.sort(key=lambda r:r[3])
    for c,dl,lag,e,a in res[:10]: print('coupling %.2f motor %2d ms pend lag %2d ms: pendulum err %.2f deg, arm vel err %.2f'%(c,dl*1e3,lag*1e3,e,a))
    print('coupling 1.0 best:', min((r for r in res if r[0]==1.0), key=lambda r:r[3])[1:4])
