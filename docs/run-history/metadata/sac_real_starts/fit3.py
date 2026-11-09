exec(open('runs/sac_real_starts/fit_swingup.py').read().split("print('nominal")[0])
for dl in (0.022,0.026,0.030,0.034): print('delay %d ms: %.2f'%(dl*1e3,err(1,1,dl)))
from pendulum_lab.controllers import load_controller
from pendulum_lab.contracts import wrap
c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path('docs/results/rig_rl_disturbance/sac_long/policy.zip'))
STATE['c']=1.0
for dl in (0.018,0.030,0.040):
    p._fixed_transport_delay_s=dl; p.exact_hanging_reset_frac=1.0
    o,_=env.reset(seed=0); p.model.dof_damping[p._pen_dof_addr]=base_damp; c.reset(); ups=[]; sat=[];over=0;prev=None
    for k in range(750):
        a=c.act(o); o,_,term,_,_=env.step(np.array([a],dtype=np.float32)); t=wrap(float(p.data.qpos[p._pen_qpos_addr])-math.pi)
        if abs(t)<0.35: ups.append(k)
        if term: break
    print('closed loop delay %d ms: first upright %.2fs, upright fraction %.0f%%, terminated=%s'%(dl*1e3, ups[0]*0.02 if ups else -1, 100*len(ups)/750, term))
