from pathlib import Path
import numpy as np
from itertools import product
import mujoco
import json
from pendulum_lab.controllers import load_controller
from pendulum_lab.environment import make_env,true_state
params=Path("configs/my_rig.json")
ckpt=Path("docs/results/sac_focused_policy.zip")
from pendulum_lab.contracts import PROFILE,observation_frame
recorded=[]
for n in (12,13,14):
    with np.load(f"runs/lqr_hw_{n}/episode_000.npz") as z:
        timing=z["hardware_timing"]
        columns={str(k):i for i,k in enumerate(z["hardware_timing_columns"])}
        mode=timing[:,columns["balance_mode"]]==1
        for i in np.flatnonzero(mode & ~np.r_[False,mode[:-1]]):
            recorded.append((n,int(timing[i,columns["sample_index"]]),None))
runs={n:np.load(f"runs/lqr_hw_{n}/episode_000.npz")["samples"] for n in (12,13,14)}
def handoff(c,n,i,lag,offset,disturbance):
    rows=runs[n]
    e=make_env(params,duration=2.)
    p=e.unwrapped
    p._fixed_transport_delay_s=lag
    e.reset(seed=0);c.reset()
    arm,theta,av,pv=rows[i,1:5]+np.array([0,*offset])
    initial=[arm,theta,av,pv]
    p.data.qpos[p._motor_qpos_addr]=arm
    p.data.qpos[p._pen_qpos_addr]=theta+np.pi
    p.data.qvel[p._motor_qvel_addr]=av
    p.data.qvel[p._pen_qvel_addr]=pv
    p.set_actuator_state(arm,planner_velocity_rad_s=av)
    planner=p._accel_transport
    for k in (3,2,1):
        activation=lag-k*PROFILE.dt
        u=float(rows[i-k,5]*PROFILE.max_accel)
        if activation <=0:planner.active_accel_rad_s2=u
        else:planner._queue.append((activation,u))
    mujoco.mj_forward(p.model,p.data)
    p.seed_measurement_ring()
    frames=[observation_frame(rows[j,1:5],rows[j-1,5]) for j in range(i-3,i)]
    frames.append(observation_frame(initial,rows[i-1,5]))
    p._obs_history.clear();p._obs_history.extend(frames)
    obs=np.concatenate(frames)
    modes=[];speeds=[];arms=[];errors=[];limited=0
    for j in range(100):
        p.data.qfrc_applied[p._pen_dof_addr] = disturbance * c.p.pendulum_inertia_kg_m2 if j<4 else 0.
        u=c.act(obs)
        modes.append(c.balancing)
        limited+=abs(u-c.pre_arm_limit_action)>.05
        obs,_,te,tr,_=e.step(np.array([u],dtype=np.float32))
        x=true_state(e)
        speeds.append(abs(x[2]));arms.append(abs(x[0]))
        errors.append(abs(x[1])<np.pi/12 and abs(x[3])<4.)
        if te or tr:break
    e.close()
    good=len(modes)==100 and all(modes[-50:]) and all(errors[-50:])
    direct=good and not any(np.diff(np.array(modes,dtype=int))<0)
    return good,direct,max(speeds),max(arms),limited

def assess_cases(c, cases):
    result=[handoff(c,n,i,lag,offset,disturbance)
            for n,i,_ in recorded for lag,(offset,disturbance) in cases]
    return {
        "simulated_handoffs":len(result),
        "held_through_final_second":sum(x[0] for x in result),
        "held_without_release":sum(x[1] for x in result),
        "median_peak_arm_speed_rad_s":float(np.median([x[2] for x in result])),
        "p90_peak_arm_deg":float(np.degrees(np.quantile([x[3] for x in result],.9))),
        "arm_limit_interventions":sum(x[4] for x in result),
    }

trials={
    "screen":list(product((.008,.02,.032),[((0.,0.,0.),0.),((.008,.12,.45),14.),((-.008,-.12,-.45),-14.)])),
    "holdout":list(product((.009,.022,.034),[((.006,-.18,.42),-18.),((-.01,.15,-.35),18.),((.013,.19,.48),10.)])),
}
outputs={}
for key in ("recenter","observed"):
    c=load_controller("lqr",params,Path("configs")/f"lqr_my_rig_{key}.json",ckpt)
    outputs[key]={name:assess_cases(c,cases) for name,cases in trials.items()}
    print(key,outputs[key],flush=True)
Path(__file__).with_name("recorded_handoffs.json").write_text(json.dumps(outputs,indent=2)+"\n")
