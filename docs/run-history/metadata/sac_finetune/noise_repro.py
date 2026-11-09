import sys, math, os, numpy as np
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path[:0]=['src','scripts']
from pathlib import Path
from multiprocessing import Pool
W2=11.30**2; TOP=2*W2
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env, ObservationNoise
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    name,alg,ck,cfg,vn,an=arg
    c=load_controller(alg,Path('configs/my_rig.json'),Path(cfg),Path(ck) if ck else None)
    base=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); p=base.unwrapped; p.exact_hanging_reset_frac=0.0
    env=ObservationNoise(base,an,vn)
    ok=0; Es=[]
    for e in range(8):
        o,_=env.reset(seed=300+e); c.reset(); th=[];w=[]
        for k in range(750):
            o,_,te,_,_=env.step(np.array([c.act(o)],dtype=np.float32))
            t=math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi); th.append(t); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if te: break
        th=np.array(th); w=np.array(w)
        ok+= len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
        Es.append(np.mean((0.5*w[150:]**2+W2*(1-np.cos(th[150:]+math.pi)))/TOP))
    return name,vn,an,ok,np.mean(Es)
if __name__=='__main__':
    CK=[('SAC current','sac','docs/results/rig_rl_disturbance/sac_long/policy.zip','configs/sac_my_rig.json'),
        ('SAC R50','sac','runs/sac_finetune/R50/policy.zip','configs/sac_my_rig.json'),
        ('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json')]
    jobs=[(*c,vn,an) for c in CK for vn,an in ((0.6,0.0),(0.7,0.0),(0.8,0.0),(0.9,0.0))]
    with Pool(15) as pool:
        for n,vn,an,ok,E in pool.map(job,jobs): print(f'{n:12s} velocity noise {vn} rad/s, angle noise {an}: pass {ok}/8 | mean energy after 3 s {E:.2f}x top')
    print('RIG: SAC 0/16, energy stays ~1.15x (spinning) | LQR 4/17')
