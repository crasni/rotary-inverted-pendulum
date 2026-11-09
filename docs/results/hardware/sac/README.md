# SAC hardware runs

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
**3 of 13 attempts passed.**
Best: [`20261109_004642`](20261109_004642/), 13.18 s continuous balance.

| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset | Checkpoint | Video | Note |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| [`20261107_163311`](20261107_163311/) | fail | 1.50 | 72.3 | duration | `sac_my_rig.json` | `rig_rl_disturbance/sac_long/policy.zip` |  |  |
| [`20261107_163340`](20261107_163340/) | fail | 0.96 | 78.0 | duration | `sac_my_rig.json` | `rig_rl_disturbance/sac_long/policy.zip` |  |  |
| [`20261107_163412`](20261107_163412/) | fail | 0.00 | 75.9 | KeyboardInterrupt | `sac_my_rig.json` | `rig_rl_disturbance/sac_long/policy.zip` |  | Stopped by hand. Pendulum reading frozen at 179.9° while the arm swung ±4 rad/s: encoder stall or pendulum held. Not a policy result. |
| [`20261107_163427`](20261107_163427/) | fail | 0.84 | 76.4 | duration | `sac_my_rig.json` | `rig_rl_disturbance/sac_long/policy.zip` |  |  |
| [`20261107_165548`](20261107_165548/) | fail | 0.82 | 80.1 | duration | `sac_my_rig.json` | `rig_rl_disturbance/sac_long/policy.zip` |  |  |
| [`20261107_165619`](20261107_165619/) | fail | 0.78 | 75.6 | duration | `sac_my_rig.json` | `rig_rl_disturbance/sac_long/policy.zip` |  |  |
| [`20261107_165745`](20261107_165745/) | fail | 0.88 | 78.8 | duration | `sac_my_rig.json` | `rig_rl_disturbance/sac_long/policy.zip` |  |  |
| [`20261107_215833`](20261107_215833/) | ✅ pass | 11.68 | 79.6 | duration | `sac_my_rig.json` | `runs/sac_finetune/P1/policy.zip` |  | First standalone SAC pass. Run used runs/sac_finetune/P1/policy.zip, promoted unchanged as rig_rl_disturbance/sac_lqr_pretrained/policy.zip (sha256 396595023ece9ba198cd9e45d9153b1bab1d76fd683ecd1dc1ca8d290fd10929). No video recorded. |
| [`20261108_211936`](20261108_211936/) | fail | 1.06 | 76.1 | duration | `sac_my_rig.json` | `runs/curriculum/versions/v6b_runner/policy.zip` |  | Report run (filmed). Reward-only SAC v6b; run used runs/curriculum/versions/v6b_runner/policy.zip, committed unchanged as rig_rl_disturbance/sac_v6b_reward_only/policy.zip (sha256 7b652fd4…80a4). Video kept by the team, not committed. |
| [`20261109_004522`](20261109_004522/) | aborted | — | — | firmware fault: RuntimeError: Firmware safety fault latched (bits=2); inspect before resetting | `sac_my_rig.json` | `runs/curriculum/versions/rnA/policy.zip` |  | Encoder fault (bits=2) latched before the start; no policy ticks. Not a policy result. |
| [`20261109_004614`](20261109_004614/) | fail | 12.78 | 78.2 | duration | `sac_my_rig.json` | `runs/curriculum/versions/rnA/policy.zip` |  | Reward-only SAC rnA, first run after re-securing the loose pendulum. Held 12.78 s. |
| [`20261109_004642`](20261109_004642/) | ✅ pass **(best)** | 13.18 | 75.1 | duration | `sac_my_rig.json` | `runs/curriculum/versions/rnA/policy.zip` |  | First official pass by reward-only SAC (rnA). Checkpoint committed unchanged as rig_rl_disturbance/sac_rnA_reward_only/policy.zip (sha256 02b6eb56…a4c5). |
| [`20261109_004713`](20261109_004713/) | ✅ pass | 13.12 | 73.7 | duration | `sac_my_rig.json` | `runs/curriculum/versions/rnA/policy.zip` |  | Report run (filmed). Reward-only SAC rnA, pass. Checkpoint committed unchanged as rig_rl_disturbance/sac_rnA_reward_only/policy.zip (sha256 02b6eb56…a4c5). Video kept by the team, not committed. |
<!-- END GENERATED -->

## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and, for newer runs,
  firmware timing and controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
