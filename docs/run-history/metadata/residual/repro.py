import sys, math, os, numpy as np
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path[:0]=['src','scripts']
from pathlib import Path
from multiprocessing import Pool
MODEL='runs/residual/residual_heldout.pt'
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env
    from pendulum_lab.residual import ResidualPendulum
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    name,alg,ck,cfg,gain=arg
    c=load_controller(alg,Path('configs/my_rig.json'),Path(cfg),Path(ck))
    base=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env=ResidualPendulum(base,MODEL,gain); p=base.unwrapped; p.exact_hanging_reset_frac=1.0
    ok=0; first=[]; caught=[]
    for e in range(8):
        o,_=env.reset(seed=900+e)
        p.data.qpos[p._pen_qpos_addr]+=np.random.default_rng(e).uniform(-0.03,0.03)  # small start variation like the rig
        c.reset(); th=[]; w=[]
        for k in range(750):
            o,_,te,_,_=env.step(np.array([c.act(o)],dtype=np.float32))
            th.append(math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi)); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if te: break
        th=np.array(th); w=np.array(w)
        ok+=len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
        k=1
        while k<len(th):
            if abs(th[k])<0.4 and abs(th[k-1])<0.4 and np.sign(th[k])!=np.sign(th[k-1]):
                j=k
                while j<len(th) and abs(th[j])<math.radians(30): j+=1
                if not first or len(first)<=e: first.append(abs(w[k]))
                caught.append(((j-k)*0.02>=2, abs(w[k])<2)); k=j+1
            else: k+=1
    slow=[c_ for c_,s_ in caught if s_]
    return name,gain,ok,np.median(first) if first else float('nan'), f'{sum(slow)}/{len(slow)}'
if __name__=='__main__':
    CK=[('SAC current','sac','docs/results/rig_rl_disturbance/sac_long/policy.zip','configs/sac_my_rig.json'),
        ('SAC A','sac','runs/sac_real_starts/A/policy.zip','configs/sac_my_rig.json'),
        ('SAC F','sac','runs/sac_real_starts/F/policy.zip','configs/sac_my_rig.json'),
        ('LQR (hw preset)','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json')]
    jobs=[(*c,g) for c in CK for g in (0.0,0.7,1.0)]
    with Pool(12) as pool:
        for n,g,ok,f,sl in pool.map(job,jobs): print(f'{n:16s} correction gain {g}: pass {ok}/8 | median first arrival {f:.1f} rad/s | slow arrivals caught {sl}')
    print('RIG for reference: SAC 0/13 pass, arrivals 1-8 rad/s, slow caught 0/24 | LQR 4/17 pass, slow caught 10/21')
