from pathlib import Path
import json
import numpy as np
from pendulum_lab.algorithms.lqr import model
from pendulum_lab.contracts import PROFILE
p,g,b,d,*_=model(Path('configs/my_rig.json'))
dry=p.pendulum_coulomb/p.pendulum_inertia_kg_m2
def always_brake(theta,omega,duration=.20):
    # Optimistic short-time calculation: the full command is available
    # immediately and the arm has no speed/travel limits or centrifugal term.
    z=np.array([abs(theta),abs(omega)])
    def diff(s):
        th,w=s
        return np.array([w,g*np.sin(th)-b*50*np.cos(th)-d*w-dry*np.tanh(w/.2)])
    h=.001
    for j in range(round(duration/h)):
        q=diff(z)
        z+=h*diff(z+.5*h*q)
        if z[1]<=0:return False
        if z[0]>=.55:return True
    return False
def load(n):
    with np.load(f'runs/lqr_hw_{n:02}/episode_000.npz') as z:
        a=z['samples'];t=z['hardware_timing'];c={str(k):i for i,k in enumerate(z['hardware_timing_columns'])}
        i=t[:,c['sample_index']].astype(int)
        return a[i],t,c
out=[]
for n in (13,14):
    x,t,c=load(n)
    mode=t[:,c['balance_mode']]==1
    losses=np.flatnonzero(np.r_[False,mode[:-1]]&~mode)
    for i in losses:
        j=np.flatnonzero(~mode[:i])[-1]+1
        ix=np.arange(max(j,i-10),i)
        raw=t[ix,c['requested_action']]*50; sent=x[ix,5]*50; v=x[ix,3];th=x[ix,2];w=x[ix,4]
        limited=(abs(raw-sent)>=20)&(raw*v>0)
        earliest=ix[np.flatnonzero(limited)[0]] if any(limited) else None
        irrecoverable=[k for k in ix if x[k,2]*x[k,4]>0 and always_brake(x[k,2],x[k,4])]
        row={
            'run':n,'release_s':round(float(x[i,0]),3),'limited_ticks_in_final_200ms':int(sum(limited)),
            'initial_limit_lead_ms':None if earliest is None else round(float((x[i,0]-x[earliest,0])*1e3)),
            'initial_limit_pendulum_deg':None if earliest is None else round(float(np.degrees(abs(x[earliest,2]))),1),
            'initial_limit_arm_speed_rad_s':None if earliest is None else round(float(abs(x[earliest,3])),2),
            'optimistic_irrecoverable_lead_ms':None if not irrecoverable else round(float((x[i,0]-x[irrecoverable[0],0])*1e3)),
        }
        print(row)
        out.append(row)
Path(__file__).with_name('releases.json').write_text(json.dumps(out,indent=2))
