exec(open('runs/sac_real_starts/fit_swingup.py').read().split("print('nominal")[0])
np.set_printoptions(precision=2, suppress=True, linewidth=180)
p._fixed_transport_delay_s=0.03; p.exact_hanging_reset_frac=1.0
for (s,a),r in list(zip(data,runs))[:3]:
    env.reset(seed=0); p.model.dof_damping[p._pen_dof_addr]=base_damp
    arm=[];th=[]
    for k in range(110):
        env.step(np.array([a[k]],dtype=np.float32)); arm.append(float(p.data.qpos[p._motor_qpos_addr])); th.append(float(p.data.qpos[p._pen_qpos_addr]))
    real_th=np.unwrap(s[1:111,1]+math.pi); real_th-=2*math.pi*np.round(real_th[0]/(2*math.pi))
    print(r[-15:])
    print('  t      ', np.arange(10,110,10)*0.02)
    print('  arm real', s[10:110:10,0]); print('  arm sim ', np.array(arm)[9:109:10])
    print('  pend real', np.degrees(real_th[9:109:10])); print('  pend sim ', np.degrees(np.array(th)[9:109:10]))
    print('  real arm |vel| max %.2f, |acc| max %.0f'%(np.abs(s[:110,2]).max(), np.abs(np.diff(s[:110,2])/0.02).max()))
