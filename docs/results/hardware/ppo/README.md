# PPO hardware runs

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
**3 of 17 attempts passed.**
Best: [`20261109_012337`](20261109_012337/), 13.52 s continuous balance.

| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset | Checkpoint | Video | Note |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| [`20261107_175250`](20261107_175250/) | fail | 1.04 | 54.3 | duration | `sac_my_rig.json` | `rig_rl_disturbance/ppo_long/policy.zip` |  | Earlier attempt, same checkpoint, not filmed. |
| [`20261108_211833`](20261108_211833/) | fail | 0.44 | 55.6 | duration | `sac_my_rig.json` | `rig_rl_disturbance/ppo_long/policy.zip` |  | Report run (filmed). Video kept by the team, not committed. |
| [`20261109_010227`](20261109_010227/) | fail | 0.18 | 53.3 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_rn/policy.zip` |  | ppo_rn (ppo_long + full rig-like reading noise). Good arrival, too-weak catch, then spun. Not adopted. |
| [`20261109_010852`](20261109_010852/) | fail | 0.00 | 64.9 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_rn05/policy.zip` |  | ppo_rn05 (half reading noise). Never caught; spun up to 37 rad/s (reward exploit). Not adopted. |
| [`20261109_011005`](20261109_011005/) | ✅ pass | 13.50 | 52.3 | duration | `sac_my_rig.json` | `rig_rl_disturbance/ppo_long/policy.zip` |  | ppo_long (previous PPO) on the repaired rig: pass from the first catch. |
| [`20261109_011116`](20261109_011116/) | fail | 0.22 | 51.6 | duration | `sac_my_rig.json` | `rig_rl_disturbance/ppo_long/policy.zip` |  | ppo_long: first catch missed, then spun (61.8 revolutions). |
| [`20261109_011145`](20261109_011145/) | fail | 0.24 | 52.4 | KeyboardInterrupt | `sac_my_rig.json` | `rig_rl_disturbance/ppo_long/policy.zip` |  | ppo_long: first catch missed, spinning; stopped by hand. |
| [`20261109_012103`](20261109_012103/) | aborted | — | — | SerialException | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecC/policy.zip` |  | Did not start: USB port not found. Not a policy result. |
| [`20261109_012112`](20261109_012112/) | fail | 0.00 | 49.1 | KeyboardInterrupt | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecC/policy.zip` |  | ppo_ecC (spin-recovery starts up to 9 rad/s). Spun; stopped by hand. |
| [`20261109_012254`](20261109_012254/) | fail | 7.84 | 50.1 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecB/policy.zip` |  | ppo_ecB. Spun; the spin was stopped by hand at about 7 s, after which it balanced. Not a valid attempt. |
| [`20261109_012337`](20261109_012337/) | ✅ pass **(best)** | 13.52 | 52.4 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecB/policy.zip` |  | ppo_ecB (energy cap + recovery starts): pass from the first catch. |
| [`20261109_012428`](20261109_012428/) | fail | 0.00 | 49.6 | KeyboardInterrupt | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecB/policy.zip` |  | ppo_ecB: first catch missed, spinning; stopped by hand. |
| [`20261109_012444`](20261109_012444/) | fail | 0.04 | 50.8 | KeyboardInterrupt | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecB/policy.zip` |  | ppo_ecB: first catch missed, spinning; stopped by hand. |
| [`20261109_013006`](20261109_013006/) | fail | 4.30 | 77.9 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecF/policy.zip` |  | ppo_ecF (+ random delay and arm lag): no runaway spin, but recovery too slow (caught at 10.7 s). |
| [`20261109_013045`](20261109_013045/) | fail | 3.84 | 75.4 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecF/policy.zip` |  | ppo_ecF: recovery too slow (caught at 11.2 s). |
| [`20261109_013444`](20261109_013444/) | fail | 2.82 | 78.4 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecG/policy.zip` |  | ppo_ecG: caught at 1.8 s, held 2.8 s, then slow recovery. |
| [`20261109_013508`](20261109_013508/) | ✅ pass | 13.14 | 65.8 | duration | `sac_my_rig.json` | `runs/curriculum/versions/ppo_ecG/policy.zip` |  | Report run. PPO ppo_ecG, pass. Checkpoint committed unchanged as rig_rl_disturbance/ppo_ecG/policy.zip (sha256 087595c5…3b2e). Video, if recorded, kept by the team. |
<!-- END GENERATED -->

## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and, for newer runs,
  firmware timing and controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
