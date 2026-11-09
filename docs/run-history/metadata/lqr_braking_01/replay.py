from itertools import product
from pathlib import Path
import json

import mujoco
import numpy as np

from pendulum_lab.contracts import PROFILE
from pendulum_lab.controllers import load_controller
from pendulum_lab.environment import make_env, true_state

params = Path("configs/my_rig.json")
from pendulum_lab.contracts import observation_frame
rows=np.load("runs/lqr_hw_11/episode_000.npz")["samples"]
checkpoint=Path("docs/results/sac_focused_policy.zip")
def near_approach(c, i, lag, perturb):
    e=make_env(params,duration=2.)
    p=e.unwrapped
    p._fixed_transport_delay_s=lag
    e.reset(seed=0);c.reset()
    initial=rows[i,1:5]+np.array([0,perturb[0],0,perturb[1]])
    arm,theta,av,pv=initial
    p.data.qpos[p._motor_qpos_addr]=arm
    p.data.qpos[p._pen_qpos_addr]=theta+np.pi
    p.data.qvel[p._motor_qvel_addr]=av
    p.data.qvel[p._pen_qvel_addr]=pv
    p.set_actuator_state(arm,planner_velocity_rad_s=av)
    planner=p._accel_transport
    for k in (3,2,1):
        activation=lag-k*PROFILE.dt
        command=float(rows[i-k,5]*PROFILE.max_accel)
        if activation <=0:
            planner.active_accel_rad_s2=command
        else:
            planner._queue.append((activation,command))
    mujoco.mj_forward(p.model,p.data)
    p.seed_measurement_ring()
    frames=[observation_frame(rows[j,1:5],rows[j-1,5]) for j in range(i-3,i)]
    frames.append(observation_frame(initial,rows[i-1,5]))
    p._obs_history.clear();p._obs_history.extend(frames)
    obs=np.concatenate(frames)
    modes=[];speeds=[];previous=theta;turns=[]
    for j in range(100):
        u=c.act(obs)
        modes.append(c.balancing)
        obs,_,te,tr,_=e.step(np.array([u],dtype=np.float32))
        x=true_state(e)
        if abs(x[1])<.16:speeds.append(abs(x[3]))
        if abs(previous)<.4 and abs(x[1])<.4 and previous*x[1]<0:turns.append(round(float(abs(x[3])),2))
        previous=x[1]
        if te or tr:break
    e.close()
    return len(modes)==100 and all(modes[-50:]),next((i for i,x in enumerate(modes) if x),None),turns

outputs={}
for key,preset in [('baseline','lqr_my_rig_learned_swing'),('braked','lqr_my_rig_braked_swing')]:
    c=load_controller('lqr',params,Path('configs')/(preset+'.json'),checkpoint)
    group={}
    for i in (116,136,138):
        outcomes=[near_approach(c,i,lag,perturb) for lag,perturb in product((.01,.02,.03),((0,0),(.01,.4),(-.01,-.4)))]
        group[str(i)]={"holds_lqr_for_final_second":sum(r[0] for r in outcomes),"scenarios":len(outcomes),
                      "nominal_20ms_first_crossing_speed_rad_s":(outcomes[3][2]+[None])[0]}
    outputs[key]=group
    print(key,group,flush=True)
Path(__file__).with_name('run11_approaches.json').write_text(json.dumps(outputs,indent=2)+'\n')
