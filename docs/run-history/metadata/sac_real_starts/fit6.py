exec(open('runs/sac_real_starts/fit_swingup.py').read().split("print('nominal")[0])
STATE['c']=1.0
def arm_err(tau, dl, n=150):
    p._fixed_transport_delay_s=dl; p._fixed_action_lag_tau_s=tau; p.exact_hanging_reset_frac=1.0
    ev=[]; ep=[]; et=[]
    for s,a in data:
        env.reset(seed=0); p.model.dof_damping[p._pen_dof_addr]=base_damp
        v=[];x=[];th=[]
        for k in range(n):
            env.step(np.array([a[k]],dtype=np.float32)); v.append(p._meas_motor_vel); x.append(float(p.data.qpos[p._motor_qpos_addr])); th.append(float(p.data.qpos[p._pen_qpos_addr]))
        ev.append(np.sqrt(np.mean((np.array(v)-s[1:n+1,2])**2))); ep.append(np.sqrt(np.mean((np.array(x)-s[1:n+1,0])**2)))
        rt=np.unwrap(s[1:91,1]+math.pi); rt-=2*math.pi*np.round(rt[0]/(2*math.pi)); et.append(np.sqrt(np.mean((np.array(th[:90])-rt)**2)))
    return np.mean(ev), np.mean(ep), np.degrees(np.mean(et))
res=[]
for tau in (0.0,0.01,0.02,0.03,0.04,0.06):
    for dl in (0.0,0.005,0.010,0.018,0.030):
        r=arm_err(tau,dl); res.append((r,tau,dl))
res.sort(key=lambda z:z[0][0])
print('nominal (tau 0, 18 ms): arm vel err %.2f rad/s, arm pos err %.3f rad, pend err %.1f deg'%arm_err(0,0.018))
for r,tau,dl in res[:8]: print('tau %2.0f ms delay %2.0f ms: arm vel err %.2f rad/s, arm pos err %.3f rad, pend err %.1f deg'%(tau*1e3,dl*1e3,*r))
