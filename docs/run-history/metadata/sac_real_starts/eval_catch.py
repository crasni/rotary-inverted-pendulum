import sys, glob, math, os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")
sys.path[:0]=['src','scripts']
from pathlib import Path
from multiprocessing import Pool
CKS={'current':'docs/results/rig_rl_disturbance/sac_long/policy.zip','A':'runs/sac_real_starts/A/policy.zip','D':'runs/sac_real_starts/D/policy.zip','H':'runs/sac_real_starts/H/policy.zip','I':'runs/sac_real_starts/I/policy.zip'}
LAGS=((0.010,0.010),(0.018,0.0),(0.008,0.020))
def env_for(dl,lag):
    from pendulum_lab.robustness import stress_env
    from train_rig_rl import RIG
    env=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':dl}); return env
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.controllers import load_controller
    import real_state_starts as R
    name,kind,dl,lag=arg
    c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path(CKS[name]))
    env=env_for(dl,lag); p=env.unwrapped
    th_of=lambda: math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi)
    if kind=='catch':   # start near top moving 3-9 rad/s; caught = upright and calm for the last 3 s of 5 s
        p.recovery_reset_frac=1.0; p.recovery_theta_max_rad=0.6; p.recovery_pen_vel_min_rad_s=3.0; p.recovery_pen_vel_max_rad_s=9.0; p.recovery_motor_vel_max_rad_s=3.0
        ok=0; n=30
        for e in range(n):
            p._pen_obs_lag_s=lag; o,_=env.reset(seed=4000+e); c.reset(); th=[]
            for k in range(250):
                o,_,te,_,_=env.step(np.array([c.act(o)],dtype=np.float32)); th.append(th_of())
                if te: break
            th=np.array(th); ok+= len(th)==250 and bool(np.all(np.abs(th[-150:])<math.radians(15)))
        return name,kind,dl,lag,f'{ok}/{n}'
    if kind=='hang':
        p.exact_hanging_reset_frac=0.0; ok=0; n=8; bal=[]
        for e in range(n):
            p._pen_obs_lag_s=lag; o,_=env.reset(seed=700+e); c.reset(); th=[]
            for k in range(750):
                o,_,te,_,_=env.step(np.array([c.act(o)],dtype=np.float32)); th.append(th_of())
                if te: break
            th=np.array(th); ok+= len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)))
        return name,kind,dl,lag,f'{ok}/{n}'
    if kind=='real':
        p._pen_obs_lag_s=lag
        H=[Path('docs/results/hardware/sac/20261107_163427'),Path('docs/results/hardware/sac/20261107_165745')]
        r=R.test('sac',Path(CKS[name]),H)
        return name,kind,dl,lag,{k:(round(v,2) if v is not None else None) for k,v in r['held_by_speed'].items()}
if __name__=='__main__':
    jobs=[(n,'catch',d,l) for n in CKS for d,l in LAGS]+[(n,'hang',d,l) for n in CKS for d,l in LAGS[:2]]+[(n,'real',0.018,0.0) for n in CKS]
    with Pool(15) as pool: res=pool.map(job,jobs)
    for name in CKS:
        row=[r for r in res if r[0]==name]
        catch=' '.join(f"{int(r[2]*1e3)}+{int(r[3]*1e3)}ms:{r[4]}" for r in row if r[1]=='catch')
        hang=' '.join(f"{int(r[2]*1e3)}+{int(r[3]*1e3)}ms:{r[4]}" for r in row if r[1]=='hang')
        real=[r[4] for r in row if r[1]=='real'][0]
        print(f"{name:8s} catch 3-9 rad/s [{catch}] | from hanging [{hang}] | held-out real states {real}")
