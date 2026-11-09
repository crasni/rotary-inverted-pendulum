# LQR hardware runs

All attempts so far used the same inputs:

| Input | File |
| --- | --- |
| Algorithm | `lqr` (balance) with SAC swing-up |
| Rig model | [`configs/my_rig.json`](../../../../configs/my_rig.json) |
| Preset | [`configs/lqr_my_rig_coupling_delay.json`](../../../../configs/lqr_my_rig_coupling_delay.json) |
| Swing-up checkpoint | [`docs/results/sac_focused_policy.zip`](../../sac_focused_policy.zip) |
| Code | git tag `lqr-hw-pass-20261107` |
| How to repeat | [LQR hardware runbook](../../../lqr-hardware-runbook.md) |

Each attempt was a 15 s hanging start. A pass needs the whole final 10 s within
15° of upright and under 4 rad/s pendulum speed. Earlier LQR presets were also
tried on hardware; those raw runs are local only and summarized in
[`configs/experiments/README.md`](../../../../configs/experiments/README.md).

[`tests/test_hardware_replay.py`](../../../../tests/test_hardware_replay.py)
replays the three passes through the current code and fails if any command
differs from what the rig was sent.

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
**4 of 18 attempts passed.**
Best: [`20261107_030202`](20261107_030202/), 12.84 s continuous balance.

| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset | Video | Note |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`20261107_022139`](20261107_022139/) | fail | 0.54 | 84.5 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_022206`](20261107_022206/) | fail | 1.10 | 74.6 | firmware fault: encoder | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_022230`](20261107_022230/) | fail | 6.94 | 76.1 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_022340`](20261107_022340/) | fail | 1.12 | 83.0 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_022408`](20261107_022408/) | fail | 4.78 | 83.1 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_022520`](20261107_022520/) | fail | 0.82 | 84.7 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_022637`](20261107_022637/) | fail | 3.36 | 83.7 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_023430`](20261107_023430/) | fail | 0.00 | 58.8 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_025142`](20261107_025142/) | fail | 1.46 | 84.5 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_025340`](20261107_025340/) | fail | 4.60 | 84.7 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_025449`](20261107_025449/) | aborted | — | — | NonFiniteFeedback | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_025738`](20261107_025738/) | ✅ pass | 11.92 | 60.7 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_025826`](20261107_025826/) | fail | 4.58 | 78.7 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_025856`](20261107_025856/) | ✅ pass | 11.10 | 81.6 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_030130`](20261107_030130/) | fail | 9.98 | 82.2 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_030202`](20261107_030202/) | ✅ pass **(best)** | 12.84 | 80.4 | duration | `lqr_my_rig_coupling_delay.json` |  | Filmed. Hardware reference run. |
| [`20261107_145456`](20261107_145456/) | fail | 4.02 | 84.2 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
| [`20261107_145555`](20261107_145555/) | ✅ pass | 10.08 | 82.1 | duration | `lqr_my_rig_coupling_delay.json` |  |  |
<!-- END GENERATED -->

## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and firmware timing and
  controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
