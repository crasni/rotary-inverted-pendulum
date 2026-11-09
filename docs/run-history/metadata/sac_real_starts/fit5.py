exec(open('runs/sac_real_starts/fit_swingup.py').read().split("print('nominal")[0])
np.set_printoptions(precision=2, suppress=True, linewidth=200)
s,a=data[0]
for dl in (0.018,0.03):
    p._fixed_transport_delay_s=dl; p.exact_hanging_reset_frac=1.0; env.reset(seed=0)
    v=[];x=[]
    for k in range(14):
        env.step(np.array([a[k]],dtype=np.float32)); v.append(float(p.data.qvel[p._motor_qvel_addr])); x.append(float(p.data.qpos[p._motor_qpos_addr]))
    print('delay',dl)
    print(' cmd*50     ', a[:14]*50)
    print(' real vel   ', s[1:15,2]); print(' sim vel    ', np.array(v))
    print(' real pos   ', s[1:15,0]); print(' sim pos    ', np.array(x))
# integrate commands ideally
iv=np.cumsum(a[:14]*50*0.02); print(' ideal vel (no delay)', iv)
# gain fit across whole run: real dv vs cmd lagged 1
for (s,a),r in zip(data,runs):
    dv=np.diff(s[:,2]); cmd=a[:-1]*50*0.02
    g=np.linalg.lstsq(np.c_[cmd[:-1]],dv[1:],rcond=None)[0][0]
    m=np.abs(s[:-2,2])<4.0
    g2=np.linalg.lstsq(np.c_[cmd[:-1][m]],dv[1:][m],rcond=None)[0][0]
    print(r[-15:], 'real dv / commanded dv (lag 1): %.3f, away from speed cap: %.3f'%(g,g2))
