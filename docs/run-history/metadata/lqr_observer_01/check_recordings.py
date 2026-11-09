"""Compare controller velocity estimators on hardware angles using measured arm forcing."""
from pathlib import Path
import json
import numpy as np

from catch_replay import runs, windows, bank, predict, metrics
from pendulum_lab.algorithms.lqr import Controller

controller=Controller(params_path=Path("configs/my_rig.json"), config={"balance_observer": True})
for n,begin,end in ((3,2.825,5.0),(9,4.205,5.5)):
    with np.load(f"runs/lqr_hw_{n:02}/episode_000.npz") as z:
        x=z["samples"]
    time=x[:,0]
    runs[n]=(x,time,(time>=begin)&(time<end))
out={}
for n in (3,9,12,13,14):
    x,time,mode=runs[n]
    win=windows((n,),"all") if n in (12,13,14) else [
        (n,i) for i in np.flatnonzero(mode)[::5] if time[i]+0.2<=time[-1]]
    data=bank(win)
    estimate=[]
    controller.reset()
    selected={i for _,i in win}
    for i,stamp in enumerate(time):
        controller.set_sample_time_us(round(stamp*1e6) % (2**32))
        controller.update_balance_observer(x[i,1:5])
        if i in selected:
            rate=controller.estimated_pendulum_velocity_rad_s
            estimate.append(x[i,4] if rate is None else rate)
    f=(1,1,0.,1)
    out[str(n)]={"windows":len(win),
         "raw_rms_deg":metrics(predict(f,win,data))[0],
         "existing_blend_rms_deg":metrics(predict(f,win,data,kind=1))[0],
         "observer_rms_deg":metrics(predict(f,win,data,offset=np.asarray(estimate)-data[3][0]))[0]}
    print(n,out[str(n)],flush=True)
Path(__file__).with_name("recordings.json").write_text(json.dumps(out,indent=2)+"\n")
