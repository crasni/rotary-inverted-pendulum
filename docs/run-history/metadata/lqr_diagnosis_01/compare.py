"""Paired ablations of the 200 ms catch predictor with recorded arm motion."""
from pathlib import Path
import json
from itertools import product
import numpy as np
from scipy.optimize import least_squares
from catch_replay import runs, windows, bank, predict, metrics, g
from causal_rate import causal_rate

train=windows((12,13),'all')
held=windows((14,),'all')
data=[bank(w) for w in (train,held)]
def measure(f,kind=0,offsets=(0.,0.)):
    return [metrics(predict(f,wins,d,kind=kind,offset=offsets[i]))
            for i,(wins,d) in enumerate(zip((train,held),data))]
fit=least_squares(lambda f:predict(f,train,data[0],0)[1:].ravel(),
                  [1,1,.008,1],bounds=([.85,.65,-.010,.2],[1.15,1.4,.035,3.]),
                  loss="soft_l1",f_scale=.07,max_nfev=40)

# Select the lag and velocity-estimation settings using only the training windows.
rank=[]
for lag, history in product((.008,.016,.020),(2,3,4)):
    correction=np.array([causal_rate(n,i,lag,history) for n,i in train])-data[0][3][0]
    for weight in (.4,.7,1.):
        score=metrics(predict((1,1,lag,1),train,data[0],offset=weight*correction))[0]
        rank.append((score,lag,history,weight))
_,lag,history,weight=min(rank)
offsets=[weight*(np.array([causal_rate(n,i,lag,history) for n,i in wins])-d[3][0])
         for wins,d in zip((train,held),data)]
output={
    "horizon_s":.2,
    "window_counts":{"training_runs_12_13":len(train),"held_out_run_14":len(held)},
    "metric_columns":["trajectory_rmse_deg","median_endpoint_error_deg","p90_endpoint_error_deg"],
    "split_order":["train","held_out"],
    "reported_velocity_and_unshifted_arm":measure((1,1,0,1)),
    "reported_velocity_and_shifted_arm":measure((1,1,lag,1)),
    "existing_velocity_blend_and_unshifted_arm":measure((1,1,0,1),kind=1),
    "refit_constants_and_lag_with_reported_velocity":{
        "factors":[float(f) for f in fit.x],"results":measure(fit.x),
    },
    "causal_rate_and_shifted_arm":{
        "shift_s":lag,"prior_intervals":history,"weight":weight,
        "results":measure((1,1,lag,1),offsets=offsets),
    },
    "retrospective_rate_and_16ms_shift":{
        "uses_future_pendulum_angle_samples_for_rate":True,
        "results":measure((1,1,.016,1),kind=3),
    },
    "unstable_time_constant_ms":1000/np.sqrt(g),
    "interpretation":"Arm forcing comes from the firmware's step counter. An effective shift can represent mechanical or sensing effects and is not a direct latency measurement. Windows from the same run can overlap. The causal-rate estimate uses only prior pendulum angles, while the diagnostic rollout uses recorded future arm motion."
}
out=Path(__file__).with_name("summary.json")
out.write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps(output,indent=2))
