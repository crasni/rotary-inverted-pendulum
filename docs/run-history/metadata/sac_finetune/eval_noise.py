import sys, os, glob, re
sys.path.insert(0,'runs/sac_finetune'); os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
import noise_repro as N
import numpy as np, math
def job(arg):
    # same as noise_repro.job but different seeds (5000+) and 10 episodes
    import torch; torch.set_num_threads(1)
    from pathlib import Path
    from pendulum_lab.robustness import stress_env, ObservationNoise
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    name,alg,ck,cfg,vn=arg
    c=load_controller(alg,Path('configs/my_rig.json'),Path(cfg),Path(ck))
    base=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); p=base.unwrapped; p.exact_hanging_reset_frac=0.0
    env=ObservationNoise(base,0.003 if vn else 0.0,vn); ok=0; Es=[]; jumps=[]
    for e in range(10):
        o,_=env.reset(seed=5000+e); c.reset(); th=[];w=[];prev=0
        for k in range(750):
            a=c.act(o); o,_,te,_,_=env.step(np.array([a],dtype=np.float32))
            t=math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi,2*math.pi); th.append(t); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if abs(t)<0.35: jumps.append(abs(a-prev)>0.5)
            prev=a
            if te: break
        th=np.array(th); w=np.array(w)
        ok+= len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
        Es.append(np.mean((0.5*w[150:]**2+N.W2*(1-np.cos(th[150:]+math.pi)))/N.TOP))
    return name,vn,ok,np.mean(Es),np.mean(jumps) if jumps else float('nan')
if __name__=='__main__':
    CK=[('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json'),('SAC current','sac','docs/results/rig_rl_disturbance/sac_long/policy.zip','configs/sac_my_rig.json')]
    for v in ('V1','V2'):
        files=sorted(glob.glob(f'runs/sac_finetune/{v}/checkpoints/*.zip'),key=lambda f:int(re.findall(r'(\d+)_steps',f)[0]))
        for f in files[1::2]: CK.append((v+' '+re.findall(r'(\d+)_steps',f)[0],'sac',f,'configs/sac_my_rig.json'))
        CK.append((f'{v} final','sac',f'runs/sac_finetune/{v}/policy.zip','configs/sac_my_rig.json'))
    with Pool(16) as pool:
        res=pool.map(job,[(*c,vn) for c in CK for vn in (0.0,0.8,1.0)])
    for c in CK:
        r={vn:(ok,E,j) for n,vn,ok,E,j in res if n==c[0]}
        print(f'{c[0]:16s} pass no-noise {r[0.0][0]}/10 | 0.8 rad/s {r[0.8][0]}/10 (energy {r[0.8][1]:.2f}x, jumps {r[0.8][2]:.0%}) | 1.0 rad/s {r[1.0][0]}/10 (energy {r[1.0][1]:.2f}x)')
