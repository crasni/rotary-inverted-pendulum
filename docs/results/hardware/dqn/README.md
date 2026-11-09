# DQN hardware runs

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
**2 of 4 attempts passed.**
Best: [`20261109_010341`](20261109_010341/), 13.56 s continuous balance.

| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset | Checkpoint | Video | Note |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| [`20261107_175326`](20261107_175326/) | fail | 0.06 | 79.7 | duration | `sac_my_rig.json` | `rig_rl_disturbance/dqn_long/policy.zip` |  | Earlier attempt, same checkpoint, not filmed. |
| [`20261108_212009`](20261108_212009/) | fail | 0.00 | 79.1 | duration | `sac_my_rig.json` | `rig_rl_disturbance/dqn_long/policy.zip` |  | Report run (filmed). Video kept by the team, not committed. |
| [`20261109_010258`](20261109_010258/) | ✅ pass | 13.06 | 79.9 | duration | `sac_my_rig.json` | `runs/curriculum/versions/dqn_rn05/policy.zip` |  | First DQN pass (dqn_rn05: dqn_long fine-tuned with rig-like reading noise). Checkpoint committed unchanged as rig_rl_disturbance/dqn_rn05/policy.zip (sha256 37be389e…939f). |
| [`20261109_010341`](20261109_010341/) | ✅ pass **(best)** | 13.56 | 79.8 | duration | `sac_my_rig.json` | `runs/curriculum/versions/dqn_rn05/policy.zip` |  | Report run (filmed). DQN dqn_rn05, pass. Checkpoint committed unchanged as rig_rl_disturbance/dqn_rn05/policy.zip (sha256 37be389e…939f). Video kept by the team, not committed. |
<!-- END GENERATED -->

## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and, for newer runs,
  firmware timing and controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
