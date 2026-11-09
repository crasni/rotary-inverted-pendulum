# Classical controller settings

LQR, MPC, iLQR and PID share one swing-up, hand-off and safety layer,
`HybridController`, a block copied identically into [`lqr.py`](../src/pendulum_lab/algorithms/lqr.py), `mpc.py`, `ilqr.py` and `pid.py`. Each algorithm
supplies only its balance law. A setting therefore means the same thing in
every algorithm, and the [hardware-tested LQR settings](../configs/lqr_my_rig_coupling_delay.json)
carry over by copying them.

A preset has this layout:

```json
{
  "arm_command_limits": { "delay_s": 0.015, "speed_rad_s": 4.7, "travel_rad": 1.42 },
  "mpc": { "...hand-off settings...": "...", "...balance gains...": "..." }
}
```

The section name (`lqr`, `mpc`, `ilqr` or `pid`) must match `--algorithm`.

## Moving the passing LQR setup to another algorithm

1. Start from the preset that reproduces the LQR pass:
   [`mpc_my_rig_transfer.json`](../configs/mpc_my_rig_transfer.json) (passed on
   hardware), [`ilqr_my_rig_lqr_match.json`](../configs/ilqr_my_rig_lqr_match.json)
   or [`pid_my_rig_lqr_match.json`](../configs/pid_my_rig_lqr_match.json).
   Their hand-off settings, arm limits, delay model and weights equal the LQR
   reference, and replaying LQR pass `030202` through them gives the same
   catch and release decisions and the same commands within 0.01 (iLQR 0.05)
   at the 99th percentile. [`tests/test_hardware_replay.py`](../tests/test_hardware_replay.py)
   fails if either property breaks. The older `ilqr_`/`pid_my_rig_transfer.json`
   presets share only the hand-off settings and were used for early attempts.
2. Tune only the **balance law** settings below in simulation.
   Compare settings with [`scripts/disturbance_sim.py`](../scripts/disturbance_sim.py),
   which adds the disturbance measured on the rig; the plain simulator passes
   almost anything. [`mpc_my_rig_low_r.json`](../configs/mpc_my_rig_low_r.json)
   is the transfer preset with `r` = 3, chosen that way. It did worse on the
   rig, so these simulators are only trusted near the tested gains
   ([Report](../Report.md#result-keep-r--30-2026-11-07)).
3. Run the same 15 s hanging-start procedure as the
   [LQR runbook](lqr-hardware-runbook.md) with `--algorithm` and `--config`
   changed.

## Arm limits (all algorithms)

| Setting | Meaning |
| --- | --- |
| `arm_command_limits.delay_s` | Command delay assumed by the limiter's stopping-distance estimate. |
| `arm_command_limits.speed_rad_s` | Arm speed the limiter aims to stay under. Must be below the 5 rad/s firmware planner limit. |
| `arm_command_limits.travel_rad` | Arm angle the limiter aims to stop before. Must be inside the hardware guard. |

The limiter runs after approach braking on every command, in swing-up and in
balance. SAC swing-up always runs limited; without a block it uses 4.7 rad/s
and 1.42 rad. The LQR reference preset uses the older spelling, top-level
`arm_speed_target_rad_s` and `arm_travel_target_rad` with the section's
`delay`. It is kept byte-for-byte because it is the recorded hardware
configuration. Do not combine both spellings.

## Hand-off settings (shared, copy unchanged)

| Setting | Default | Meaning |
| --- | --- | --- |
| `swing_up` | energy law | `"sac"` uses the `--checkpoint` policy for swing-up and recovery. |
| `approach_brake_speed_rad_s` | off | SAC only. Limits swing-up commands so the pendulum crosses upright at about this speed. |
| `capture_angle`, `capture_speed` | 0.5 rad, 2 rad/s | Pendulum must be inside both to start balancing. |
| `capture_arm_angle`, `capture_arm_speed` | no limit | Arm must also be inside both. |
| `capture_max_action` | no limit | Reject a capture whose first balance command would exceed this. |
| `release_angle` | 0.55 rad | Return to swing-up beyond this pendulum angle. |
| `balance_velocity_blend` | 0 | Share of pendulum speed taken from a 3-point angle derivative instead of the firmware speed. |
| `balance_observer` | false | After capture, replace pendulum speed with a model-based estimate from recent samples and firmware timestamps. |
| `arm_recenter_rate_rad_s` | 0 | Start balancing around the capture arm angle and move the reference to center at this rate. |
| `arm_recenter_max_rad` | 0.64 | Largest starting reference for recentering. |
| `balance_arm_speed_headroom_rad_s` | 3.6 | Recentering and the observer pause above this arm speed. |

## Model settings

| Setting | Algorithms | Meaning |
| --- | --- | --- |
| `delay` | all four | Arm command delay in the delay-augmented model. LQR defaults to 25 ms and always uses the delayed model; MPC, iLQR and PID use it only when set. iLQR then adds its nonlinear terms to that model; PID predicts the state until the commands in flight have acted and runs its loop on that prediction. |
| `pendulum_input_delay_s` | all four | Separate, longer delay for the pendulum's response to the arm. |
| `balance_coupling_scale` | all four | Scales the arm-to-pendulum coupling `b` in the balance model. |
| `balance_state_lead_s` | MPC, iLQR, PID | Predicts the state up to one tick ahead under the last command. Use instead of the delayed model, not with it. |

## Balance law settings (tune per algorithm)

| Algorithm | Settings |
| --- | --- |
| LQR | `q` (4 weights: arm, pendulum, arm speed, pendulum speed), `r` |
| MPC | `q`, `r`, `horizon`, `max_iterations`, `solver_ftol`, `arm_constraint_rad` (default 1.6), `arm_speed_constraint_rad_s` (default 4.8), `capture_r` (input weight used only for the `capture_max_action` check; default `r`) |
| iLQR | `q`, `r`, `horizon`, `max_iterations` |
| PID | `kp`, `kd`, `ki`, `arm_kp`, `arm_kd`. In `pid_my_rig_lqr_match.json`, `kp`, `kd`, `arm_kp` and `arm_kd` are the LQR gain expressed on the predicted state (exact, see the [Report](../Report.md#matching-pid-and-ilqr-to-the-lqr-pass)); `ki` = 10 adds a small integral on the angle error. |

Set MPC's arm constraints to match `arm_command_limits`, as the transfer preset
does. Otherwise MPC plans motions the limiter will cut.

## Energy swing-up settings

These apply only without `"swing_up": "sac"`, or as the mixed-in part with
`swing_energy_mix`: `energy_gain`, `swing_arm_kp`, `swing_arm_kd`,
`swing_target_energy_ratio`, `swing_drive_cap`, `swing_energy_max_ratio`
(adaptive target) and `swing_energy_mix` (SAC only, 0 to 1).
The fixed thresholds behind the energy law, adaptive target, braking and
observer are named constants at the top of
the shared hand-off block in [`lqr.py`](../src/pendulum_lab/algorithms/lqr.py).
