# MPC hardware runs

**First pass: [`20261107_155613`](20261107_155613/)**, 10.66 s of continuous
balance (caught at 4.23 s and held to the end; final 10 s within 5.89° and
1.33 rad/s). Write-up: [Report](../../../../Report.md#mpc-hardware-pass-2026-11-07).

The passing run used:

| Input | File |
| --- | --- |
| Algorithm | `mpc` (balance) with SAC swing-up |
| Rig model | [`configs/my_rig.json`](../../../../configs/my_rig.json) |
| Preset | [`configs/mpc_my_rig_transfer.json`](../../../../configs/mpc_my_rig_transfer.json) |
| Swing-up checkpoint | [`docs/results/sac_focused_policy.zip`](../../sac_focused_policy.zip) |
| Code | git tag `mpc-hw-pass-20261107` |

To repeat it, follow the [LQR runbook](../../../lqr-hardware-runbook.md) with
`--algorithm mpc --config configs/mpc_my_rig_transfer.json`. Earlier attempts
used other presets (see the Preset column); only the transfer preset passed.

<!-- BEGIN GENERATED: scripts/promote_hardware_run.py -->
**1 of 28 attempts passed.**
Best: [`20261107_155613`](20261107_155613/), 10.66 s continuous balance.

| Run | Result | Longest balance (s) | Max arm (°) | Stop | Preset | Video | Note |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`20261107_150201`](20261107_150201/) | fail | 2.80 | 81.1 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_150408`](20261107_150408/) | fail | 0.98 | 76.8 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_150624`](20261107_150624/) | fail | 0.00 | 16.5 | firmware fault: encoder | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_150642`](20261107_150642/) | aborted | — | — | firmware fault: encoder | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_150748`](20261107_150748/) | fail | 1.14 | 84.7 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_150833`](20261107_150833/) | fail | 0.56 | 84.4 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_151046`](20261107_151046/) | fail | 3.10 | 84.4 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_151727`](20261107_151727/) | fail | 2.02 | 84.5 | duration | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_151807`](20261107_151807/) | aborted | — | — | RuntimeError | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_151820`](20261107_151820/) | fail | 2.32 | 84.2 | duration | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_151856`](20261107_151856/) | fail | 1.08 | 84.8 | duration | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_151926`](20261107_151926/) | fail | 5.90 | 78.9 | duration | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_152007`](20261107_152007/) | fail | 0.00 | 54.0 | KeyboardInterrupt | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_152024`](20261107_152024/) | fail | 1.68 | 75.3 | duration | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_152120`](20261107_152120/) | fail | 1.52 | 84.3 | duration | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_152222`](20261107_152222/) | fail | 0.78 | 82.6 | duration | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_152353`](20261107_152353/) | aborted | — | — | firmware fault: RuntimeError: Firmware safety fault latched (bits=2); inspect before resetting | `mpc_my_rig_strict_catch.json` |  |  |
| [`20261107_153810`](20261107_153810/) | fail | 0.44 | 78.1 | duration | `mpc_my_rig_low_r.json` |  |  |
| [`20261107_153838`](20261107_153838/) | fail | 0.40 | 84.4 | duration | `mpc_my_rig_low_r.json` |  |  |
| [`20261107_153932`](20261107_153932/) | fail | 0.20 | 73.6 | duration | `mpc_my_rig_low_r.json` |  |  |
| [`20261107_154431`](20261107_154431/) | fail | 0.90 | 84.7 | duration | `mpc_my_rig_low_r.json` |  |  |
| [`20261107_154510`](20261107_154510/) | fail | 1.40 | 81.3 | duration | `mpc_my_rig_low_r.json` |  |  |
| [`20261107_154540`](20261107_154540/) | fail | 1.12 | 84.4 | duration | `mpc_my_rig_low_r.json` |  |  |
| [`20261107_155316`](20261107_155316/) | aborted | — | — | RuntimeError | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_155325`](20261107_155325/) | fail | 1.42 | 80.6 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_155419`](20261107_155419/) | fail | 2.60 | 84.8 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_155533`](20261107_155533/) | fail | 1.38 | 84.6 | duration | `mpc_my_rig_transfer.json` |  |  |
| [`20261107_155613`](20261107_155613/) | ✅ pass **(best)** | 10.66 | 64.5 | duration | `mpc_my_rig_transfer.json` |  | First MPC hardware pass. Preset mpc_my_rig_transfer.json (same commands as the LQR pass). |
<!-- END GENERATED -->

## Files in each run

- `run.json`: the exact command, hardware profile and rig-model hash.
- `summary.json`: the graded result.
- `episode_000.npz`: the 50 Hz trace (`samples`) and, for newer runs,
  firmware timing and controller internals (`hardware_timing`).
- `video.*` and `note.md`, when added with `--video` and `--note`.
