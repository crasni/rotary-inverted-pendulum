# Reproducing the LQR hardware pass

Use this runbook for Group 8’s assembled Furuta pendulum. On 2026-11-07 the
setup below passed **3 of 16** attempts. The best, run `030202`, held **12.84 s
of continuous balance** and is the filmed run. The other passes, `025738` and
`025856`, held 11.92 s and 11.10 s. Plan for repeats. All 16 attempts are
committed in [`docs/results/hardware/lqr/`](results/hardware/lqr/),
and the code is git tag `lqr-hw-pass-20261107`.

## Passing setup

| Item | Use |
| --- | --- |
| Algorithm | `lqr` |
| Rig model | [`configs/my_rig.json`](../configs/my_rig.json) |
| Controller preset | [`configs/lqr_my_rig_coupling_delay.json`](../configs/lqr_my_rig_coupling_delay.json) |
| Swing-up checkpoint | [`docs/results/sac_focused_policy.zip`](results/sac_focused_policy.zip) |
| Balance model | 15 ms effective arm delay and 25 ms effective pendulum response delay |
| Hardware profile | Protocol 3, 32 microsteps, 500000 baud, 50 Hz control |
| Start and duration | Arm physically centered, pendulum hanging still, 15 s |

SAC provides swing-up. LQR takes over near upright and performs the balance.
The `calibrated` and `swing` presets are separate experiments. Use the
`coupling_delay` file above for the reference procedure.

## Before powering the rig

1. From the repository root, use the existing `.venv`. For a fresh setup,
   follow the [install instructions](../README.md#install), including the RL
   dependencies for loading the checkpoint. Use the [deployment guide](deployment.md)
   to commission a freshly assembled rig or flashed board.
2. If you restored the files or changed the software, run the simulation smoke
   test below. It starts hanging and uses the same three inputs as hardware.
3. With **12 V off**, verify that the pendulum is secure at the pivot and
   swings freely. Physically center the arm and let the pendulum hang still.
   Close serial monitors and clear the full swing area. If the port has changed,
   run `.venv/bin/pendulum-lab ports` and substitute the listed port.

```bash
.venv/bin/pendulum-lab sim \
  --algorithm lqr \
  --params configs/my_rig.json \
  --config configs/lqr_my_rig_coupling_delay.json \
  --checkpoint docs/results/sac_focused_policy.zip \
  --start hanging --seed 0 --duration 15 \
  --output "runs/lqr_coupling_delay_sim_$(date +%Y%m%d_%H%M%S)"
```

## Run the 15-second trial

From the repository root, start with **12 V off**. If you need video evidence,
start recording before powering the rig so the video shows the hanging start
and the whole run.

```bash
run_dir="runs/lqr_coupling_delay_$(date +%Y%m%d_%H%M%S)"
.venv/bin/pendulum-lab hardware \
  --algorithm lqr \
  --params configs/my_rig.json \
  --config configs/lqr_my_rig_coupling_delay.json \
  --checkpoint docs/results/sac_focused_policy.zip \
  --port /dev/ttyUSB0 --duration 15 \
  --output "$run_dir"
```

At the power prompt, switch on 12 V, keep clear, and type `RUN`. Stay by the
power switch during the trial. Switch off 12 V when the command ends, before
touching or recentering the arm.

## Check the result

In the same shell, run `cat "$run_dir/summary.json"`. The graded pass has
`"stop_reason": "duration"`, `"samples": 750`, and `"success": true`.
Every sample in the final 10 seconds must have pendulum angle error at most
15° and pendulum speed at most 4 rad/s. `longest_balance_s` describes the
longest streak anywhere in the run. Use `success` to check the final window.

In the best run (`030202`), LQR took over at 2.09 s and held through shutdown.
The final 10 seconds stayed within 7.82° of upright and 3.16 rad/s. Arm travel
peaked at 80.4° during swing-up and stayed within 22.7° during balance.

## If a repeat fails

| What you see | What to do |
| --- | --- |
| The full trial finishes with `success: false` | Save the trace. With power off, recenter and let the pendulum settle before a same-preset repeat. Keep the duration at 15 s for a comparable grading attempt. |
| It catches, falls, and catches again | Save `episode_000.npz`. Compare `balance_mode`, `pre_arm_limit_action`, the sent `action`, and arm speed around the drop. In the first repeat after the pass, the limiter cut a −50 rad/s² request to nearly zero at the speed limit. |
| The pendulum loosens, detaches, or the arm skips | Switch off 12 V and inspect the assembly before another powered attempt. The arm log reports step counts, so physical skipping can escape the log. |
| The runner stops on a feedback or firmware fault | Keep the stop reason and trace. Follow the [fault guidance](deployment.md#verify-the-motor) before powering again. |

If the pendulum was reattached in the same geometry and swings freely, start
with the saved model. If the pivot, weight position, or free-swing behavior
changed, validate the model against fresh data before refitting. See the
[SysID guide](system-identification.md#independent-validation).

## Preserve the evidence

Keep `run.json`, `summary.json`, `episode_000.npz`, and any video together.
To submit a run, pass or fail, copy it into the repository and update the
indexes with one command:

```bash
.venv/bin/python scripts/promote_hardware_run.py "$run_dir" --video path/to/video.mp4 --note "filmed"
```

It lands in `docs/results/hardware/lqr/` (see [hardware results](results/hardware/README.md)).
Use `run.json` for the invocation and hardware profile, `summary.json` for the
graded result, and `episode_000.npz` for the trace. Use unique output directories
for every attempt. Git ignores `runs/`, so anything left only there is not
submitted.

The local bundle `runs/lqr_hardware_pass_20261107_025738.tar.gz` is not
committed. It contains the 025738 logs, rig model, preset, checkpoint, and a snapshot of
the Python and firmware source files available after the pass. It also records
the current dependency versions for reference. Extract into an empty directory
and follow the install steps above. Add any separately recorded video to your
submission evidence.

To verify the three control inputs from the repository root:

```bash
sha256sum --check <<'CHECKSUMS'
14d0eab5d939b8fbae869114473ea37e68382706417d076ff1db4ecb0dd4e80f  configs/my_rig.json
569dc057befda97bd5f29b44da5f6630a68ead349555876f1a2df06508a5f3fb  configs/lqr_my_rig_coupling_delay.json
93f85de5e9cce382fc7a4f2a8ff22791e3942a40bf3810f42fcc4eb66c1a5356  docs/results/sac_focused_policy.zip
CHECKSUMS
```
