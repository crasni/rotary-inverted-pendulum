# dqn_rn05 lineage (DQN, official passes 2026-11-09)

Every step is Stable-Baselines3 DQN with its own discrete action set
(`policy.zip.actions.json`) and the lab's reward. Simulator only: no rig data,
no imitation, no other algorithm's weights or data.

| # | Checkpoint | Trained by |
|---|---|---|
| 1 | `runs/gpu_rl_20261016/dqn_train` | DQN in simulation (`configs/dqn_gpu.json`) |
| 2 | `runs/disturbance_rl_v2/dqn` | DQN with random pendulum disturbances |
| 3 | `docs/results/rig_rl_disturbance/dqn_long` | DQN, disturbance RMS 0-12 (failed on the rig: 0.06 s, 0 s) |
| 4 | **`dqn_rn05`** (this folder) | DQN, 400k steps (`train_rig_rl.py dqn --rig-noise 0.5 --delay-range 0.006 0.02 --disturbance-rms 6`) |

Step 4 adds rig-like heavy-tailed reading noise on the pendulum angle and speed
(Student-t, 0.5 x the rig's measured 0.6 deg / 0.6 rad/s rms;
`train_smooth_sac.RigReadingNoise`) to the training environment. It is the same
environment change that made reward-only SAC (rnA) pass; the learner is DQN's own.
Full rig noise (1.0x) made DQN fail the standard simulator, so 0.5x was used.
policy.zip sha256: 37be389e831b834e92ddee59cb2fe7726fd4fdd527dd23a87e89d4bcb7c7939f (copied unchanged from runs/curriculum/versions/dqn_rn05).
