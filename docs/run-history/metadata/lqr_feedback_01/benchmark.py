from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import product
import json
import runpy
import mujoco
import numpy as np
from pendulum_lab.algorithms.lqr import Controller
from pendulum_lab.environment import make_env,true_state
from pendulum_lab.evaluation import summarize
params=Path("configs/my_rig.json")
base=json.loads(params.read_text())
cfg=json.loads(Path("configs/lqr_my_rig_robust.json").read_text())["lqr"]
sample_rig=runpy.run_path("scripts/robustness.py")["sample_rig"]
states=[np.load(f"runs/lqr_hw_{n:02}/episode_000.npz")["samples"][0,1:5] for n in range(1,9)]

from collections import deque
from pendulum_lab.plant.pendulum_env import PENDULUM_LSB_RAD
def run(env,settings,seed=0,start=None,lag=0.):
    c=Controller(params_path=params,config=settings)
    p=env.unwrapped
    p._meas_ring=deque(maxlen=40)
    obs,_=env.reset(seed=seed)
    if start is not None:
        arm,theta,av,pv=start
        p.data.qpos[p._motor_qpos_addr]=arm
        p.data.qpos[p._pen_qpos_addr]=theta+np.pi
        p.data.qvel[p._motor_qvel_addr]=av
        p.data.qvel[p._pen_qvel_addr]=pv
        p.set_actuator_state(arm,planner_velocity_rad_s=av)
        mujoco.mj_forward(p.model,p.data)
        p.seed_measurement_ring()
        p._obs_history.clear()
        obs=p._obs()
    history=deque([obs[-6:]]*4,maxlen=4)
    samples=[]
    peak_v=0.
    rng=np.random.default_rng(seed+5678)
    clipped=0
    release_count=0
    for i in range(750):
        latest=obs[-6:].copy()
        if lag:
            ring=p._meas_ring
            stale=round((p._obs_staleness_s+lag)/p.model.opt.timestep)
            new=max(0,len(ring)-1-stale)
            past=max(0,new-p._meas_win_substeps)
            angle=round(ring[new][1]/PENDULUM_LSB_RAD)*PENDULUM_LSB_RAD
            old_angle=round(ring[past][1]/PENDULUM_LSB_RAD)*PENDULUM_LSB_RAD
            pv=(angle-old_angle)/(max(1,new-past)*p.model.opt.timestep)
            latest[1:3]=np.sin(angle-np.pi),np.cos(angle-np.pi)
            latest[4]=pv
        if lag:
            latest[4]+=rng.normal(0.,.30)
        history.append(latest)
        was=c.balancing
        action=c.act(np.concatenate(history))
        if was and not c.balancing: release_count+=1
        if c.balancing and abs(action)>.999:clipped+=1
        obs,_,te,tr,_=env.step(np.array([action],dtype=np.float32))
        x=true_state(env)
        peak_v=max(peak_v,abs(x[2]))
        samples.append([(i+1)*.02,*x,action])
        if te or tr:break
    result=summarize(samples,'arm_guard' if te else 'duration',15)
    return result['success'],te,release_count,clipped,result['longest_balance_s'],result['max_arm_deg'],peak_v
def assess(candidates,seed,count,lag_mode):
    rng=np.random.default_rng(seed)
    lag_rng=np.random.default_rng(seed+87121)
    scores={key:[] for key,c in candidates}
    with TemporaryDirectory() as directory:
        path=Path(directory)/"rig.json"
        for i in range(count):
            doc,f=sample_rig(base,rng)
            path.write_text(json.dumps(doc))
            e=make_env(path)
            e.unwrapped._fixed_transport_delay_s=f["delay_s"]
            lag=0 if lag_mode=="none" else lag_rng.uniform(.004,.012)
            for key,c in candidates:
                scores[key].append(run(e,c,seed+i,start=states[i%len(states)],lag=lag))
            e.close()
    return {key:(sum(x[0] for x in rows),sum(x[1] for x in rows),sum(x[2] for x in rows),round(float(np.mean([x[3] for x in rows])),1),round(max(x[5] for x in rows),1)) for key,rows in scores.items()}


if __name__ == "__main__":
    candidates = [(name, json.loads(Path(path).read_text())["lqr"]) for name, path in (
        ("15ms", "configs/lqr_my_rig_adaptive_15ms.json"),
        ("filtered", "configs/lqr_my_rig_filtered.json"),
    )]
    report = {
        "description": "Paired perturbed MuJoCo rigs with recorded hardware starts. Extra feedback stress adds 4–12 ms relative pendulum sensor lag and 0.30 rad/s RMS pendulum velocity noise.",
        "metrics": ["passes", "arm_guard_stops", "releases", "mean_clipped_balance_commands", "max_arm_deg"],
        "episodes_per_batch": 24,
        "results": {},
    }
    for seed in (50196, 69561, 71319, 80589):
        results = assess(candidates, seed, 24, "extra")
        report["results"][str(seed)] = results
        print(seed, results, flush=True)
    Path(__file__).with_name("summary.json").write_text(json.dumps(report, indent=2) + "\n")
