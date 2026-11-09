import sys, math, numpy as np
sys.path[:0]=['src','scripts']
from pathlib import Path
import gymnasium as gym
from pendulum_lab.robustness import stress_env
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import wrap
from pendulum_lab.evaluation import summarize
from train_rig_rl import RIG

class ArmDrift(gym.Wrapper):
    """Add AR(1) acceleration error to what the arm executes; the policy still sees its own commands."""
    def __init__(self, env, rms, corr=0.95, gain=(1.0,1.0)):
        super().__init__(env); self.rms=rms; self.corr=corr; self.gain=gain; self.rng=np.random.default_rng()
    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng=np.random.default_rng(seed+23)
        self.value=self.rng.normal()*self.rms; self.g=self.rng.uniform(*self.gain)
        obs,info=self.env.reset(seed=seed,options=options); self.sent=[float(f) for f in obs[5::6]]; return obs,info
    def step(self, action):
        a=float(np.asarray(action).item())
        self.value=self.corr*self.value+math.sqrt(1-self.corr**2)*self.rng.normal()*self.rms
        obs,r,te,tr,info=self.env.step(np.array([np.clip(a*self.g+self.value,-1,1)],dtype=np.float32))
        self.sent=self.sent[1:]+[a]; obs=obs.copy(); obs[5::6]=self.sent
        return obs,r,te,tr,info

def run(ck, rms, delay, eps=8, gain=(1.0,1.0)):
    c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path(ck))
    base=stress_env(Path('configs/my_rig.json'),{**RIG,'delay_s':delay}); env=ArmDrift(base,rms,gain=gain); p=base.unwrapped
    out=[]
    for e in range(eps):
        p.exact_hanging_reset_frac=1.0; p.upright_reset_frac=0.0
        o,_=env.reset(seed=500+e); c.reset(); rows=[]; reason='duration'; sat=[]
        for k in range(750):
            a=c.act(o); o,_,te,tr,info=env.step(np.array([a],dtype=np.float32))
            th=wrap(float(p.data.qpos[p._pen_qpos_addr])-math.pi)
            rows.append([(k+1)*0.02,float(p.data.qpos[p._motor_qpos_addr]),th,float(p.data.qvel[p._motor_qvel_addr]),float(p.data.qvel[p._pen_qvel_addr]),a])
            if abs(th)<0.35: sat.append(abs(a)>0.95)
            if te: reason='guard'; break
        out.append((summarize(rows,reason,15), np.mean(sat) if sat else np.nan))
    return out
if __name__=='__main__':
    for rms in (0.0,0.03,0.06,0.1):
        for delay in (0.018,0.03):
            o=run('docs/results/rig_rl_disturbance/sac_long/policy.zip',rms,delay)
            print('drift rms %.2f (%.1f rad/s^2) delay %d ms: pass %d/8, longest balance %.1fs, full strength upright %.0f%%, guard stops %d'%(rms,rms*50,delay*1e3,sum(r['success'] for r,_ in o),np.mean([r['longest_balance_s'] for r,_ in o]),100*np.nanmean([s for _,s in o]),sum(r['stop_reason']!='duration' for r,_ in o)))
