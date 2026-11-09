# PID hardware runs

**First pass: [`20261107_161650`](20261107_161650/)**, 10.42 s of continuous
balance (caught at 4.29 s and held to the end; final 10 s within 2.38° and
0.95 rad/s, arm within 20°). Write-up: [Report](../../../../Report.md#pid-hardware-pass-2026-11-07).

The passing run used:

| Input | File |
| --- | --- |
| Algorithm | `pid` (balance) with SAC swing-up |
| Rig model | [`configs/my_rig.json`](../../../../configs/my_rig.json) |
| Preset | [`configs/pid_my_rig_lqr_match.json`](../../../../configs/pid_my_rig_lqr_match.json) |
| Swing-up checkpoint | [`docs/results/sac_focused_policy.zip`](../../sac_focused_policy.zip) |
| Code | git tag `pid-hw-pass-20261107` |

To repeat it, follow the [LQR runbook](../../../lqr-hardware-runbook.md) with
`--algorithm pid --config configs/pid_my_rig_lqr_match.json`. The earlier
`pid_my_rig_transfer.json` attempt did not pass.

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
**1 of 8 attempts passed.**
Best: [`20261107_161650`](20261107_161650/), 10.42 s continuous balance.

| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset | Video | Note |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`20261107_152516`](20261107_152516/) | fail | 1.24 | 78.6 | duration | `pid_my_rig_transfer.json` |  |  |
| [`20261107_160634`](20261107_160634/) | fail | 0.50 | 80.7 | duration | `pid_my_rig_lqr_match.json` |  |  |
| [`20261107_160701`](20261107_160701/) | fail | 2.76 | 79.5 | duration | `pid_my_rig_lqr_match.json` |  |  |
| [`20261107_160734`](20261107_160734/) | fail | 0.50 | 83.8 | duration | `pid_my_rig_lqr_match.json` |  |  |
| [`20261107_161502`](20261107_161502/) | fail | 6.20 | 73.4 | duration | `pid_my_rig_lqr_match.json` |  |  |
| [`20261107_161536`](20261107_161536/) | fail | 8.26 | 77.5 | duration | `pid_my_rig_lqr_match.json` |  |  |
| [`20261107_161611`](20261107_161611/) | fail | 0.56 | 81.8 | duration | `pid_my_rig_lqr_match.json` |  |  |
| [`20261107_161650`](20261107_161650/) | ✅ pass **(best)** | 10.42 | 72.5 | duration | `pid_my_rig_lqr_match.json` |  | First PID hardware pass. Preset pid_my_rig_lqr_match.json (delay predictor, gains from the LQR solution, Ki = 10). |
<!-- END GENERATED -->

## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and, for newer runs,
  firmware timing and controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
