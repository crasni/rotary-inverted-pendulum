import sys, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
import mujoco, torch
torch.set_num_threads(1)
from pendulum_lab.robustness import stress_env
from pendulum_lab.residual import ResidualPendulum
from train_rig_rl import RIG
import real_state_starts as R
blob=torch.load('runs/residual/residual_heldout.pt',weights_only=False); HELD=blob['held_out_runs']
def tru(s,k):
    u=np.unwrap(s[k-1:k+2,1]); a=s[k-1:k+2,0]; return s[k,0],s[k,1],(a[2]-a[0])/0.04,(u[2]-u[0])/0.04
for gain in (0.0,0.7,1.0,1.3):
    base=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':0.010}); env=ResidualPendulum(base,'runs/residual/residual_heldout.pt',gain); env.reset(seed=0); p=base.unwrapped
    E={2:[],5:[],10:[]}; EU={2:[],5:[],10:[]}
    for r in HELD:
        s,a=R.load_run(r)
        for k in range(10,len(s)-12,3):
            R.set_state(base,s,a,k-3); env.actions=[a[k-4],a[k-5],a[k-6],a[k-7]]
            for j in (k-3,k-2,k-1): env.step(np.array([a[j]],dtype=np.float32))
            arm,th,armv,penv=tru(s,k); p.data.qpos[p._pen_qpos_addr]=th+math.pi; p.data.qvel[p._pen_qvel_addr]=penv; mujoco.mj_forward(p.model,p.data)
            for j in range(10):
                env.step(np.array([a[k+j]],dtype=np.float32))
                if j+1 in E:
                    e=math.remainder(float(p.data.qpos[p._pen_qpos_addr])-math.pi-s[k+j+1,1],2*math.pi)
                    E[j+1].append(e)
                    if abs(th)<0.5: EU[j+1].append(e)
    f=lambda L: math.degrees(np.sqrt(np.mean(np.square(L))))
    print(f'gain {gain}: held-out runs theta error at 40/100/200 ms  all {f(E[2]):.2f}/{f(E[5]):.1f}/{f(E[10]):.1f} deg | near upright {f(EU[2]):.2f}/{f(EU[5]):.1f}/{f(EU[10]):.1f} deg')
