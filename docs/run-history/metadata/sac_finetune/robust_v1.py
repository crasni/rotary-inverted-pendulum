import sys, os, math, numpy as np
sys.path[:0]=['src','scripts','runs/sac_finetune']; os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from multiprocessing import Pool
import gymnasium as gym
class SpeedNoise(gym.Wrapper):
    """Noise that grows with pendulum speed (like the rig): std = a + b*|speed|; plus occasional 3-LSB spikes."""
    def __init__(s,env,a,b,spike=0.0): super().__init__(env); s.a=a; s.b=b; s.spike=spike; s.rng=np.random.default_rng(); s.h=[]
    def reset(s,*,seed=None,options=None):
        o,i=s.env.reset(seed=seed,options=options); s.rng=np.random.default_rng((seed or 0)+99); f=s.f(o); s.h=[f]*4; return np.concatenate(s.h),i
    def f(s,o):
        x=np.asarray(o[-6:],dtype=np.float32).copy(); w=x[4]; x[4]+=s.rng.normal()*(s.a+s.b*abs(w))
        if s.spike and s.rng.random()<s.spike: x[4]+=s.rng.choice([-1,1])*0.57
        return x
    def step(s,a):
        o,r,te,tr,i=s.env.step(a); s.h=s.h[1:]+[s.f(o)]; return np.concatenate(s.h),r,te,tr,i
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env, ObservationNoise, varied_parameters, CASES
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    import tempfile
    name,alg,ck,cfg,case=arg
    c=load_controller(alg,Path('configs/my_rig.json'),Path(cfg),Path(ck))
    tmp=tempfile.mkdtemp()
    extra={'combined':CASES['combined'],'inertia_high':CASES['inertia_high'],'friction_high':CASES['friction_high']}.get(case,{})
    params=varied_parameters(Path('configs/my_rig.json'),extra,Path(tmp)/'p.json') if extra else Path('configs/my_rig.json')
    delay={'delay18_n08':0.018,'delay30_n08':0.030}.get(case,0.010)
    base=stress_env(params,{**RIG,**{k:v for k,v in extra.items() if k in('delay_s','max_accel','speed_limit')},'delay_s':extra.get('delay_s',delay)}); p=base.unwrapped; p.exact_hanging_reset_frac=0.0
    if case=='noise1.4': env=ObservationNoise(base,0.003,1.4)
    elif case in('delay18_n08','delay30_n08','combined','inertia_high','friction_high'): env=ObservationNoise(base,0.003,0.8)
    elif case=='speed_noise': env=SpeedNoise(base,0.2,0.06)
    elif case=='spiky': env=SpeedNoise(base,0.4,0.03,spike=0.1)
    else: env=base
    ok=0
    for e in range(10):
        o,_=env.reset(seed=7000+e); c.reset(); th=[];w=[]
        for k in range(750):
            o,_,te,_,_=env.step(np.array([c.act(o)],dtype=np.float32))
            th.append(math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi)); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if te: break
        th=np.array(th); w=np.array(w); ok+= len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
    return name,case,ok
if __name__=='__main__':
    CK=[('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json'),('SAC current','sac','docs/results/rig_rl_disturbance/sac_long/policy.zip','configs/sac_my_rig.json'),
        ('V1 final','sac','runs/sac_finetune/V1/policy.zip','configs/sac_my_rig.json'),('V2 final','sac','runs/sac_finetune/V2/policy.zip','configs/sac_my_rig.json')]
    cases=['noise1.4','delay18_n08','delay30_n08','combined','inertia_high','friction_high','speed_noise','spiky']
    with Pool(16) as pool: res=pool.map(job,[(*c,k) for c in CK for k in cases])
    print('case'.ljust(14),' '.join(n[0].ljust(12) for n in CK))
    for k in cases: print(k.ljust(14),' '.join((str(next(o for n,cc,o in res if n==c[0] and cc==k))+'/10').ljust(12) for c in CK))
