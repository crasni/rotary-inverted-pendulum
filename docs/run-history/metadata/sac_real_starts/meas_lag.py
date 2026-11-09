exec(open('runs/sac_real_starts/fit_swingup.py').read().split("print('nominal")[0])
import glob
runs2=[r for r in sorted(glob.glob('docs/results/hardware/sac/2026*'))+sorted(glob.glob('runs/sac_A_*')) if '163412' not in r]
data2=[R.load_run(r) for r in runs2]
N=90
def traces(dl):
    p._fixed_transport_delay_s=dl; p.exact_hanging_reset_frac=1.0; out=[]
    for s,a in data2:
        env.reset(seed=0); p.model.dof_damping[p._pen_dof_addr]=base_damp
        sub=[]; arm=[]
        for k in range(N):
            env.step(np.array([a[k]],dtype=np.float32)); sub.append(float(p.data.qpos[p._pen_qpos_addr])); arm.append(p._meas_motor_vel)
        rt=np.unwrap(s[1:N+1,1]+math.pi); rt-=2*math.pi*np.round(rt[0]/(2*math.pi))
        out.append((np.array(sub),rt,np.array(arm),s[1:N+1,2]))
    return out
for dl in (0.005,0.010,0.018):
    tr=traces(dl)
    armerr=np.mean([np.sqrt(np.mean((a-ra)**2)) for _,_,a,ra in tr])
    for shift_ms in (0,10,20,30):
        f=shift_ms/20.0; errs=[]
        for sim,real,_,_ in tr:
            t=np.arange(N); simd=np.interp(t-f,t,sim)  # sim pendulum seen 'shift' later
            errs.append(np.sqrt(np.mean((simd-real)**2)))
        print('actuator delay %2d ms (arm vel err %.2f) + pendulum reading late by %2d ms: pend err %.1f deg'%(dl*1e3,armerr,shift_ms,np.degrees(np.mean(errs))))
