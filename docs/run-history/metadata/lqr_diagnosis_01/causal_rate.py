from catch_replay import *
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def causal_rate(n,i,lag,history=3):
    x,t,m=runs[n]
    angle=np.unwrap(x[:,2])
    prior=slice(i-history,i+1)
    ts=t[prior]-t[i]
    # Make a spline of the arm samples that were available by t[i].
    start=max(0,i-history-4)
    spline=UnivariateSpline(t[start:i+1],x[start:i+1,1],s=(i+1-start)*1e-6)
    arm_acc=spline.derivative(2)
    arm_v=spline.derivative(1)
    def dynamics(s, state):
        theta,w=state
        v=arm_v(t[i]+s-lag)
        u=arm_acc(t[i]+s-lag)
        return w,g*np.sin(theta)+b*np.cos(theta)*u+v*v*np.sin(theta)*np.cos(theta)-d*w-dry*np.tanh(w/.2)
    raw=x[i,4]
    def backwards(w):
        return solve_ivp(dynamics,(0,ts[0]),(angle[i],w),t_eval=ts[::-1],max_step=.004,rtol=1e-6,atol=1e-7).y[0,::-1]
    base=backwards(raw)
    jac=backwards(raw+1)-base
    return raw+float(jac @ (angle[prior]-base)/(jac@jac))
