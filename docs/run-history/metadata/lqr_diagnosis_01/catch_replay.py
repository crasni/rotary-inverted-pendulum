from pathlib import Path
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from pendulum_lab.algorithms.lqr import model
from pendulum_lab.contracts import wrap
P=Path('configs/my_rig.json')
p,g,b,d,*_=model(P)
dry=p.pendulum_coulomb/p.pendulum_inertia_kg_m2
def loads(run):
    with np.load(f'runs/lqr_hw_{run:02}/episode_000.npz') as z:
        x=z['samples'].copy()
        tm=z['hardware_timing']; c={str(k):i for i,k in enumerate(z['hardware_timing_columns'])}
        x=x[tm[:,c['sample_index']].astype(int)]
        read=(tm[:,c['host_read_start_s']]+tm[:,c['host_read_end_s']])/2
        dev=tm[:,c['device_time_us']]*1e-6
        # Map the independent firmware clock onto host seconds using the full run.
        ts=dev-dev[0]
        t=ts*np.polyfit(ts,read,1)[0]
        mode=tm[:,c['balance_mode']]==1
    return x,t,mode
runs={i:loads(i) for i in (12,13,14)}
duration=.20; h=.0025; steps=round(duration/h)
tau=np.arange(2*steps+1)*h/2
sample_step=4
def windows(ids, mode):
    out=[]
    for n in ids:
        x,t,m=runs[n]
        switch=np.flatnonzero(m&~np.r_[False,m[:-1]])
        for j in switch:
            for k in ([j] if mode=='handoff' else [j,j+5,j+10,j+15,j+20]):
                if k>=len(t) or k<5 or not m[k] or t[k]+duration>t[-1]:continue
                out.append((n,k))
    return out
def bank(wins, smooth=.001):
    arms={n:UnivariateSpline(t,x[:,1],s=len(t)*smooth**2) for n,(x,t,m) in runs.items()}
    Y=[];initial=[];vel=[]
    for n,i in wins:
        x,t,m=runs[n]; angle=np.unwrap(x[:,2]); arm=arms[n]
        t0=t[i]; obs_t=t0+tau[::2*sample_step]
        raw=x[i,4]
        diff0=(angle[i]-angle[i-1])/(t[i]-t[i-1])
        backward=np.polyfit(t[i-2:i+1]-t0,angle[i-2:i+1],2)[1]
        symmetric=np.polyfit(t[i-2:i+3]-t0,angle[i-2:i+3],2)[1]
        Y.append(np.interp(obs_t,t,angle))
        initial.append(angle[i]);vel.append([raw,.55*raw+.45*backward,backward,symmetric,diff0])
    return arms,np.array(Y).T,np.array(initial),np.array(vel).T
def predict(f,wins,data,kind=0,offset=0.):
    gravity,coupling,lag,drag=f
    arms,Y,theta,omega=data
    x=np.stack([theta,omega[kind]+offset])
    t=np.array([runs[n][1][i] for n,i in wins])
    v=np.stack([arms[n].derivative(1)(tau+t0-lag) for (n,i),t0 in zip(wins,t)],axis=1)
    acc=np.stack([arms[n].derivative(2)(tau+t0-lag) for (n,i),t0 in zip(wins,t)],axis=1)
    def diff(z,k):
        angle,rate=z
        sin,cos=np.sin(angle),np.cos(angle)
        return np.stack([rate,gravity*g*sin+coupling*b*cos*acc[k]+v[k]**2*sin*cos-drag*d*rate-dry*np.tanh(rate/.2)])
    trace=[x[0]]
    for i in range(steps):
        k=2*i
        q=diff(x,k)
        r=diff(x+h*q/2,k+1)
        s=diff(x+h*r/2,k+1)
        u=diff(x+h*s,k+2)
        x=x+h*(q+2*r+2*s+u)/6
        if (i+1)%sample_step==0:trace.append(x[0])
    return wrap(np.asarray(trace)-Y)
def metrics(x):
    rms=np.degrees(np.sqrt(np.mean(x[1:]**2)))
    ends=np.degrees(abs(x[-1]))
    return [round(float(rms),2),round(float(np.median(ends)),2),round(float(np.quantile(ends,.9)),2)]
if __name__=='__main__':
    train=windows((12,13),'all'); hold=windows([14],'all')
    to=windows((12,13),'handoff');ho=windows([14],'handoff')
    print('window counts',len(train),len(hold),len(to),len(ho),flush=True)
    for smooth in (.0006,.001,.0015):
        dt,dh,dto,dho=(bank(w,smooth) for w in (train,hold,to,ho))
        print('SMOOTH',smooth,flush=True)
        for kind in range(5):
            for lag in (0.,.008,.016):
                f=[1,1,lag,1]
                print(kind,lag,[metrics(predict(f,w,data,kind)) for w,data in zip((train,hold,to,ho),(dt,dh,dto,dho))],flush=True)
            fit=least_squares(lambda f:predict(f,train,dt,kind)[1:].ravel(),[1,1,.008,1],
                              bounds=([.85,.65,-.010,.2],[1.15,1.4,.035,3.]),loss='soft_l1',f_scale=.07, max_nfev=40)
            print('FIT',kind,np.round(fit.x,4),[metrics(predict(fit.x,w,data,kind)) for w,data in zip((train,hold,to,ho),(dt,dh,dto,dho))],flush=True)
