import sys, math; sys.path[:0]=['src']
import numpy as np
from pathlib import Path
from pendulum_lab.rig_session import RigSession, SimClient, save_episode
from pendulum_lab.controllers import load_controller
OUT = Path('/tmp/claude-1000/-home-crasni--claude-spares-spare-5XYn5e/2c54c7e6-41fd-405e-a0e2-05f63dd4d333/scratchpad')
t=[0.0]
def sleep(d): t[0]+=max(d,0)
clock=lambda: t[0]
ck = sys.argv[1] if len(sys.argv) > 1 else 'runs/sac_finetune/V1/policy.zip'
for seed in range(3):
    client=SimClient(Path('configs/my_rig.json'), seed=seed)
    c=load_controller('sac',Path('configs/my_rig.json'),Path('configs/sac_my_rig.json'),Path(ck))
    sess=RigSession(client, arm_limits=(0.018,4.4,1.35), episode_seconds=12, clock=clock, sleep=sleep)
    sess.start(confirm=lambda _: 'RUN')
    try:
        for ep in range(4):
            c.reset()
            e=sess.run_episode(lambda o: (c.act(o), c.requested_action))
            r=save_episode(OUT, ep, e)
            rt=sess.reset()
            print(f'seed {seed} ep {ep}: longest {r["longest_balance_s"]:.2f}s max arm {r["max_arm_deg"]:.0f} | reset took {rt:.1f}s', flush=True)
    finally:
        sess.stop()
