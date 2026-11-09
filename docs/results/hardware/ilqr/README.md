# iLQR hardware runs

**First pass: [`20261107_161026`](20261107_161026/)**, 13.26 s of continuous
balance (caught at 1.67 s and held to the end; final 10 s within 1.49° and
0.95 rad/s, arm within 23°). Write-up: [Report](../../../../Report.md#ilqr-hardware-pass-2026-11-07).

The passing run used:

| Input | File |
| --- | --- |
| Algorithm | `ilqr` (balance) with SAC swing-up |
| Rig model | [`configs/my_rig.json`](../../../../configs/my_rig.json) |
| Preset | [`configs/ilqr_my_rig_lqr_match.json`](../../../../configs/ilqr_my_rig_lqr_match.json) |
| Swing-up checkpoint | [`docs/results/sac_focused_policy.zip`](../../sac_focused_policy.zip) |
| Code | git tag `ilqr-hw-pass-20261107` |

To repeat it, follow the [LQR runbook](../../../lqr-hardware-runbook.md) with
`--algorithm ilqr --config configs/ilqr_my_rig_lqr_match.json`. The earlier
`ilqr_my_rig_transfer.json` attempts did not pass.

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
**1 of 10 attempts passed.**
Best: [`20261107_161026`](20261107_161026/), 13.26 s continuous balance.

| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset | Video | Note |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`20261107_151300`](20261107_151300/) | fail | 4.30 | 79.5 | duration | `ilqr_my_rig_transfer.json` |  |  |
| [`20261107_151344`](20261107_151344/) | fail | 1.18 | 82.0 | firmware fault: encoder | `ilqr_my_rig_transfer.json` |  |  |
| [`20261107_151433`](20261107_151433/) | fail | 1.32 | 84.5 | duration | `ilqr_my_rig_transfer.json` |  |  |
| [`20261107_151547`](20261107_151547/) | fail | 1.28 | 80.7 | duration | `ilqr_my_rig_transfer.json` |  |  |
| [`20261107_152428`](20261107_152428/) | fail | 0.82 | 84.6 | duration | `ilqr_my_rig_transfer.json` |  |  |
| [`20261107_160807`](20261107_160807/) | fail | 1.34 | 81.7 | duration | `ilqr_my_rig_lqr_match.json` |  |  |
| [`20261107_160844`](20261107_160844/) | fail | 6.54 | 83.4 | duration | `ilqr_my_rig_lqr_match.json` |  |  |
| [`20261107_160921`](20261107_160921/) | fail | 0.94 | 83.5 | duration | `ilqr_my_rig_lqr_match.json` |  |  |
| [`20261107_160955`](20261107_160955/) | fail | 6.46 | 79.2 | duration | `ilqr_my_rig_lqr_match.json` |  |  |
| [`20261107_161026`](20261107_161026/) | ✅ pass **(best)** | 13.26 | 47.1 | duration | `ilqr_my_rig_lqr_match.json` |  | First iLQR hardware pass. Preset ilqr_my_rig_lqr_match.json (LQR delay model plus nonlinear terms). |
<!-- END GENERATED -->

## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and, for newer runs,
  firmware timing and controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
