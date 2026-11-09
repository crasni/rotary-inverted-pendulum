from pathlib import Path
import numpy as np
from pendulum_lab.algorithms.lqr import model
p,a,b,d,*_=model(Path("configs/my_rig.json"))
dry=p.pendulum_coulomb/p.pendulum_inertia_kg_m2
def traces(paths):
    ans=[]
    for path in paths:
        with np.load(path) as z:
            x=z["samples"]; dev=z["device_time_us"]*1e-6
            t=dev-dev[0]; t*=np.polyfit(t,x[:,0],1)[0]
        ans.append((t,x[:,1],np.unwrap(x[:,4])-np.pi,x[:,5],x[:,6]))
    return ans
def rows(traces,lag,length):
    target=[];drive=[]
    for t,phase,theta,av,pv in traces:
        shifted=np.interp(t-lag,t,av)
        for i in range(len(t)):
            k=np.searchsorted(t,t[i]+length)
            if k>=len(t) or phase[i]<0 or phase[k]!=phase[i] or t[i]-lag<t[0]:continue
            sl=slice(i,k+1)
            drift=a*np.sin(theta[sl])+av[sl]**2*np.sin(theta[sl])*np.cos(theta[sl])-d*pv[sl]-dry*np.tanh(pv[sl]/.2)
            y=pv[k]-pv[i]-np.trapezoid(drift,t[sl])
            cv=np.cos(.5*(theta[i:k]+theta[i+1:k+1]))
            target.append(y)
            drive.append(np.dot(cv,np.diff(shifted[sl])))
    return np.asarray(target),np.asarray(drive)
train=traces(sorted(Path('runs/envelope_lqr_01').glob('*.npz')))
hold=traces(sorted(Path('runs/envelope_02').glob('*.npz')))
for length in (.020,.032,.044):
    candidates=[]
    for lag in np.arange(-.008,.031,.001):
        y,x=rows(train,lag,length)
        gain=x@y/(x@x)
        err=np.sqrt(np.mean((y-gain*x)**2))
        candidates.append((err,lag,gain,len(x)))
    best=min(candidates)
    near=sorted(candidates,key=lambda x:abs(x[1]))[0]
    scored=[]
    for e,lag,gain,count in (near,best):
        y,x=rows(hold,lag,length)
        scored.append((round(float(lag)*1e3,1),round(float(gain),3),round(float(e),3),round(float(np.sqrt(np.mean((y-gain*x)**2))),3),count,len(x)))
    print('window',length,'[lag ms, coupling, train RMSE, hold RMSE, counts]',scored,flush=True)
