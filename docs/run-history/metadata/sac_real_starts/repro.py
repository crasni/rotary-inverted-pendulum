import sys, math, numpy as np
sys.path[:0]=['src','scripts','runs/sac_real_starts']
from pathlib import Path
from multiprocessing import Pool
def job(args):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env
    from pendulum_lab.controllers import load_controller
    from pendulum_lab.contracts import state_from_obs
    from train_rig_rl import RIG
    name, ck, dl, lag = args
    c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path(ck))
    env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':dl}); p=env.unwrapped; p.exact_hanging_reset_frac=0.0
    first=[];ok=0;bal=[]
    for e in range(6):
        o,_=env.reset(seed=700+e); p._pen_obs_lag_s=lag; p.seed_measurement_ring(); o=p._obs(); c.reset(); th=[];w=[]
        for k in range(750):
            o,_,te,_,_=env.step(np.array([c.act(o)],dtype=np.float32))
            th.append(math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi)); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if te: break
        th=np.array(th); w=np.array(w)
        cr=[abs(w[k]) for k in range(1,len(th)) if abs(th[k])<0.4 and np.sign(th[k])!=np.sign(th[k-1])]
        first.append(cr[0] if cr else np.nan); ok+= len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)))
        bal.append(np.mean(np.abs(th)<math.radians(15)))
    return name, dl, lag, ok, np.nanmedian(first), np.mean(bal)
if __name__=='__main__':
    cks={'current':'docs/results/rig_rl_disturbance/sac_long/policy.zip','A':'runs/sac_real_starts/A/policy.zip','D':'runs/sac_real_starts/D/policy.zip'}
    jobs=[(n,c,dl,lag) for n,c in cks.items() for dl,lag in ((0.018,0.0),(0.010,0.010),(0.008,0.020))]
    with Pool(9) as pool:
        for n,dl,lag,ok,f,b in pool.map(job,jobs):
            print(f'{n:8s} motor {dl*1e3:2.0f} ms + pend lag {lag*1e3:2.0f} ms: pass {ok}/6, first top-crossing speed {f:.1f} rad/s, balanced {b:.0%}')
