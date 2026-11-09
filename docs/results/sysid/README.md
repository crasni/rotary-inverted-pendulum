# System identification data

Raw recordings and outputs of the supplied workflow
([guide](../../system-identification.md)), copied unchanged from the local
`runs/` folder, which git ignores. The fitted rig model they produced is
[`configs/my_rig.json`](../../../configs/my_rig.json)
(sha256 `14d0eab5…4e80f`), which every hardware result uses. The video of the
process is added separately (see [MATERIALS.md](../../../MATERIALS.md)).

| Folder | Recorded | Contents | Result |
| --- | --- | --- | --- |
| [`sysid_train_01/`](sysid_train_01/) | 2026-11-06 21:11 | Tare (`tare.npz`), three 10 s free swings at 100 Hz (`free_run_*.npz`), `metadata.json`, and the simulator comparison for run 1 (`freeswing_compare.*`) | Used to fit `configs/my_rig.json`. Comparison angle RMSE 1.39 rad over 10 s (see limitations) |
| [`sysid_holdout_01/`](sysid_holdout_01/) | 2026-11-06 21:15 | Tare and three more free swings, not used for fitting | Held out for validation |
| [`sysid_validation_01/`](sysid_validation_01/) | after fitting | Comparison of the fitted model with each held-out swing (`validation.json`, per-run CSV/JSON/PNG) | Angle RMSE 0.31, 0.28 and 0.23 rad; parameters not refitted |
| [`sysid_motor_envelope_20261107_025411/`](sysid_motor_envelope_20261107_025411/) | 2026-11-07 02:54 | Powered motor sweep (`v2p5_a30.npz`) and `summary.json` | Passed at 2.5 rad/s and 30 rad/s²; peak measured 3.25 rad/s |

Fitted values (from `configs/my_rig.json`): small-amplitude natural frequency
12.27 rad/s, measured pendulum inertia 4.65 × 10⁻⁵ kg·m², viscous friction
1.02 × 10⁻⁵ N·m·s, Coulomb friction 9.4 × 10⁻⁶ N·m.

## Driven-response recordings (motor powered)

From the same identification evening, in [`driven/`](driven/). Fixed command
patterns with the pendulum hanging, recorded at about 200 Hz.

| Folder | Recorded | What it is | Notes |
| --- | --- | --- | --- |
| [`envelope_01/`](driven/envelope_01/) | 2026-11-06 21:22 | Motor-envelope sweep, 1.5 rad/s at 25 rad/s² | Source of the coupling fit below |
| [`coupling_01.json`](driven/coupling_01.json) | 2026-11-06 21:22 | `sysid fit-coupling` on `envelope_01` | Coupling gain 0.93 of the model, R² 0.86 (67 samples) |
| [`envelope_lqr_01/`](driven/envelope_lqr_01/) | 2026-11-06 21:28 | Sweeps at 1.5, 2.5, 3.5 and 4.5 rad/s, all at 50 rad/s² | Arm up to 5.1 rad/s, pendulum swings up to 75°: the swing-up range |
| [`envelope_02/`](driven/envelope_02/) | 2026-11-07 01:28 | Sweep at 4.5 rad/s, 50 rad/s² | Arm up to 5.1 rad/s |
| [`accel_step_real_008/`](driven/accel_step_real_008/) (+ `_sim_008`) | 2026-11-06 21:18 | Acceleration step response, scale 0.08 (pulses up to 4 rad/s²) | Complete; arm RMSE vs simulator 0.054 rad |
| [`accel_chirp_real_003/`](driven/accel_chirp_real_003/) (+ `_sim_003`) | 2026-11-06 21:21 | 0.5–5.5 Hz chirp, scale 0.003 | Complete; very small excitation |

## Earlier session

[`earlier_session_20261106_2110/`](earlier_session_20261106_2110/) is a backup
taken at 21:10 on 2026-11-06, before the final fit. It holds an earlier
identification (its own free swings, holdout and validation), its fitted
`my_rig.json` (sha256 `d3823068…`), and the first step and chirp attempts (one
step capture is marked incomplete). Its fit differs from the final model
mainly in friction: Coulomb 8.7 × 10⁻⁷ against 9.4 × 10⁻⁶ N·m, viscous
1.23 × 10⁻⁵ against 1.02 × 10⁻⁵ N·m·s.

## Limitations

- Free-swing comparisons accumulate phase error over 10 s, so the training
  comparison (1.39 rad) is much worse than the held-out ones; neither was used
  to refit parameters.
- These experiments do not identify transport delay or motor torque. The
  delays used by the controllers (15 ms arm, 25 ms pendulum) come from
  hardware balance data, and the [Report](../../../Report.md#why-the-simulator-was-optimistic)
  compares them with the balance-time response.
