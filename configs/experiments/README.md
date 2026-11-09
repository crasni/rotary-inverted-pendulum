# Experiment presets (history)

These presets led to the passing LQR setup or were earlier transfer attempts.
Hardware outcomes come from the local `runs/` folders, which are not committed;
only the passing preset's attempts are in `docs/results/hardware/`.
They still load and are kept because [Report.md](../../Report.md) and
[docs/algorithms.md](../../docs/algorithms.md) cite them. For new runs use the
presets in [`configs/`](../): the LQR hardware reference
[`lqr_my_rig_coupling_delay.json`](../lqr_my_rig_coupling_delay.json) or a
`*_my_rig_transfer.json` preset.

## LQR, in the order they were developed

| Preset | What it added | Outcome |
| --- | --- | --- |
| `lqr_my_rig.json` | First LQR on the identified rig, energy swing-up | Hardware lqr_hw_01–02: no hold |
| `lqr_my_rig_capture.json` | Tighter capture angle (0.4 rad) | Hardware lqr_hw_03–04: best hold 2.24 s |
| `lqr_my_rig_guarded.json` | Arm angle/speed capture gate, `capture_max_action`, R = 30 | Hardware lqr_hw_05–06: best hold 0.16 s |
| `lqr_my_rig_robust.json` | Swing drive cap, stronger arm centering, wider capture | Hardware lqr_hw_07–08: no hold |
| `lqr_my_rig_adaptive.json` | Adaptive swing energy target | Hardware lqr_hw_09: 1.18 s hold |
| `lqr_my_rig_adaptive_15ms.json` | Adaptive, with a 15 ms delay model | Simulation only |
| `lqr_my_rig_filtered.json` | Adaptive + velocity blend 0.45 | Hardware lqr_hw_10: 0.08 s hold |
| `lqr_my_rig_learned_swing.json` | SAC swing-up, LQR balance | Hardware lqr_hw_11 (3 s): crossed upright too fast |
| `lqr_my_rig_braked_swing.json` | + approach braking at 4.5 rad/s | Hardware lqr_hw_12–14: best hold 1.18 s |
| `lqr_my_rig_recenter.json` | + arm recentering at 0.35 rad/s | Simulation only |
| `lqr_my_rig_observed.json` | + velocity observer | Hardware lqr_hw_15, 17, 18: best hold 2.36 s |
| `lqr_my_rig_calibrated.json` | + 27 ms pendulum delay, coupling × 1.1 | Hardware, 5 attempts: best hold 6.26 s |
| `lqr_my_rig_swing.json` | Calibrated + 20 % energy law mixed into SAC swing-up | Hardware, 4 attempts: best hold 2.18 s |
| **[`../lqr_my_rig_coupling_delay.json`](../lqr_my_rig_coupling_delay.json)** | Observed + 25 ms pendulum delay | **Passed 3 of 16 hardware attempts** |

## Earlier transfer presets

| Preset | Notes |
| --- | --- |
| `pid_my_rig.json`, `ilqr_my_rig.json`, `mpc_my_rig.json` | SAC swing-up with a one-tick state prediction. Hand-off settings differ from the LQR pass. MPC hardware: 3 attempts, best hold 1.14 s. |
| `mpc_my_rig_coupling_delay.json` | MPC with the 15/25 ms delay model. Hardware runs 031705 and 031732 held 2.54 s and 1.96 s. |
| `mpc_my_rig_strict_catch.json` | MPC transfer preset with a stricter catch gate (arm speed 2.5 rad/s, first command 0.7). Hardware 2026-11-07: 0 of 10, best hold 5.9 s. Replaced by `mpc_my_rig_low_r.json`. |

The `*_my_rig_transfer.json` presets replace these. They copy the passing LQR
hand-off settings exactly.
