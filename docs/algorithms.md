# Algorithm development before hardware is ready

All seven required files contain initial controller implementations for tuning.
The optional `custom.py` exposes the learned-swing-up experiment separately.
No dependencies were added to `pyproject.toml`.

## Model and conventions

Use x = [arm angle, theta from upright, arm velocity, theta velocity].
The input v is normalized; physical arm acceleration u = 50 v.
With mass m, COM distance l, pivot inertia J, arm radius r, and viscous friction d,
the acceleration-driven pendulum approximation is

```
arm_ddot = u
theta_ddot = (m*g*l/J)*sin(theta) + (m*r*l/J)*cos(theta)*u
             + arm_dot**2*sin(theta)*cos(theta) - (d/J)*theta_dot
```

For the supplied hinge orientation, the coupling at upright is positive.
Linearizing sin(theta) = theta and cos(theta) = 1 gives

```
A = [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, m*g*l/J, 0, -d/J]]
B = 50 * [[0], [0], [1], [m*r*l/J]]
```

The implementations discretize with a zero-order hold at 0.02 s. This reduced
model assumes prescribed arm acceleration. It omits Coulomb friction, motor
tracking error, measurement quantization, and command delay; the supplied MuJoCo
environment includes those effects. Thus analytic stability alone does not
establish simulator or hardware performance. Validate mass, geometry, friction,
and delay against your assembled rig.

## Classical controllers

Each file contains its own helpers, as required by the repository convention.
The swing-up, capture, state-estimation and arm-limit code is one block that
`lqr.py`, `mpc.py`, `ilqr.py` and `pid.py` each carry an identical copy of.
Edit it in `lqr.py` and run `scripts/sync_handoff.py`; a test fails if the
copies drift. The [settings reference](controller-settings.md) lists every key.
All use energy shaping outside the upright capture region. The normalized
pendulum energy is E/J = theta_dot^2/2 + (m*g*l/J)*cos(theta). Acceleration pumps
the energy error with theta_dot*cos(theta), with arm centering, damping, and a
small alternating kick to leave exact hanging rest. A braking heuristic reduces
outward motion near the arm boundary. Capture/release use angle hysteresis and a
capture velocity threshold. This swing-up needs empirical tuning.

- PID: outer arm position/velocity loop requests a small lean; inner pendulum
  PID commands acceleration, with bounded integral and saturation anti-windup.
- LQR: discrete Riccati solution and full-state feedback near upright.
- MPC: bounded linear finite-horizon optimization, predicted arm travel/speed
  constraints, Riccati terminal cost, and shifted warm starts. Failed optimization
  falls back to LQR; inspect failure frequency before hardware use.
- iLQR: nonlinear midpoint dynamics, finite-difference Jacobians, backward
  feedforward/feedback recursion, and forward line search with bounded controls.
  This baseline clips controls rather than solving a box-constrained backward pass.

Configure each through a `pid`, `lqr`, `mpc`, or `ilqr` object in the CLI JSON.
Common keys: `q` (four diagonal costs), `r`, `energy_gain`, `capture_angle`,
`release_angle`, `capture_speed`, `swing_arm_kp`, `swing_arm_kd`, `horizon`, and `max_iterations` where applicable.
PID also accepts `kp`, `ki`, `kd`, `arm_kp`, and `arm_kd`.
MPC and iLQR solve online: measure worst-case runtime before a 50 Hz hardware run.

### Adaptive LQR swing-up for the identified rig

The pendulum energy per unit pivot inertia is measured in s⁻², with radians
treated as dimensionless. Dividing by `a = m*g*l/J` gives the dimensionless
energy `e = cos(theta) + theta_dot²/(2*a)`. The preset in
`configs/experiments/lqr_my_rig_adaptive.json` starts with an energy target of 1.02.
When the pendulum turns too far from upright, the controller adds up to
`1 - cos(theta_closest)` for the next excursion. If it passes near upright
with excess speed, the controller lowers the target using the kinetic term.
The range is 0.98 to 1.18, and each update is limited to 0.10. The balance
handoff still requires a feasible first acceleration command.

The filtered preset blends the firmware's 8 ms pendulum velocity estimate
with a three-point backward derivative of the host's angle samples at 50 Hz.
The derivative uses wrapped angle increments and the preset gives it 45%
weight. The blended rate feeds LQR and its first-command check. The raw rate
still feeds swing-up and the capture speed limit.

The [observer preset](../configs/experiments/lqr_my_rig_observed.json) uses recent angle
changes and arm-speed increments to estimate pendulum velocity during
balance. For each prior timestamp `t0`, it uses the relation
`theta(t) - theta(t0) = (t - t0)*omega(t) - integral((s - t0)*alpha(s), ds)`.
Here `alpha` includes gravity, damping, centrifugal acceleration, and the
pendulum impulse implied by the measured arm-speed change. It fits one
velocity to the four preceding intervals using the firmware timestamps on
hardware. It blends that estimate with the reported velocity, reduces its
weight when the angle history fits poorly, and bounds the correction at
0.7 rad/s. It applies the estimate after handoff while the pendulum is within
0.30 rad of upright and the arm speed is below 3.6 rad/s. The capture gate
and the first LQR command still use the earlier feedback calculation.

For the 65 mm arm, the 50 rad/s² command limit gives 3.25 m/s² of tangential
pivot acceleration, or about 0.33 g. In the reduced model with a stationary
arm and pendulum, full inward acceleration cancels gravity at about 18°
from upright. Inward momentum and pending commands also affect capture.
The acceleration bound and the switching condition address separate parts
of the swing-up problem. See
[Åström and Furuta](https://www.control.lth.se/fileadmin/control/Education/EngineeringProgram/FRTN10/2020/AstromFuruta2000.pdf)
and the full rotary dynamics in
[Cazzolato and Prime](https://digital.library.adelaide.edu.au/items/2dee399a-84e2-4990-a1e1-c191c44d3b92).

### Graded LQR with learned swing-up and arm braking headroom

`lqr` loads the specified SAC checkpoint when the configuration selects
`swing_up: sac`. It uses SAC during swing-up and recovery after a missed catch.
The Riccati feedback supplies the balance commands after the LQR capture gate
accepts the state. Both phases use the same arm command limiter.
For each direction it estimates the speed that fits within the remaining arm
travel using a reaction time and a braking distance. It also targets an arm
speed below the planner's 5 rad/s limit so the controller retains room to
accelerate. The limits in `configs/custom_my_rig.json` are controller targets.
The hardware runner and firmware enforce the lab envelope.

```bash
pendulum-lab sim --algorithm lqr --checkpoint docs/results/sac_focused_policy.zip \
  --params configs/my_rig.json --config configs/experiments/lqr_my_rig_learned_swing.json \
  --start hanging --duration 15 --output runs/lqr_sim_learned_check
```

## Learning controllers

PPO and SAC output continuous normalized acceleration. DQN uses a discrete action
grid (default -1, -0.5, 0, 0.5, 1). No learned controller has a swing-up helper.
Training uses Stable-Baselines3, seeded runs, Monitor episode-return CSV logs,
periodic checkpoints, and a final `policy.zip`. Hyperparameters passed to SB3
belong inside the algorithm's JSON object. General settings are `seed`,
`total_steps`, `learning_rate`, `device`, and `checkpoint_every`.
DQN's top-level `actions` grid is persisted as `.zip.actions.json`; keep this
sidecar beside both final and intermediate checkpoints.

The save/load and monitoring workflow follows the
[official SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html).
Short training round trips verify the pipeline; they do not produce successful
policies. Compare multiple seeds and evaluate hanging starts on held-out seeds.

## Transferring the rig settings

**Current presets.** [`mpc_my_rig_transfer.json`](../configs/mpc_my_rig_transfer.json),
[`ilqr_my_rig_transfer.json`](../configs/ilqr_my_rig_transfer.json) and
[`pid_my_rig_transfer.json`](../configs/pid_my_rig_transfer.json) copy the
hand-off settings and arm limits of the [passing LQR preset](lqr-hardware-runbook.md)
unchanged and keep their own balance gains. Tune only the balance gains; see
[controller settings](controller-settings.md#moving-the-passing-lqr-setup-to-another-algorithm).
Simulation results are in [Report.md](../Report.md#transferring-the-lqr-setup-to-mpc-ilqr-and-pid).

```bash
.venv/bin/pendulum-lab sim --algorithm mpc \
  --params configs/my_rig.json --config configs/mpc_my_rig_transfer.json \
  --checkpoint docs/results/sac_focused_policy.zip --start hanging --duration 15 \
  --output "runs/mpc_transfer_sim_$(date +%Y%m%d_%H%M%S)"
```

### Earlier transfer presets (history)

The presets below predate the shared hand-off and are kept in
`configs/experiments/` for the results that cite them.

The `arm_command_limits` setting applies the same speed and stopping-distance
calculation in simulation and hardware. The underlying policy still supplies
the control request at every tick. This optional command limit complements the
hardware runner and firmware guards.

| Algorithm | Preset | Checkpoint | How it uses the checkpoint |
| --- | --- | --- | --- |
| SAC | [`sac_my_rig.json`](../configs/sac_my_rig.json) | `docs/results/sac_randomized_policy.zip` | SAC supplies both swing-up and balance. |
| PID | [`pid_my_rig.json`](../configs/experiments/pid_my_rig.json) | `docs/results/sac_focused_policy.zip` | SAC swings up. PID balances with a 12 ms state prediction. |
| MPC | [`mpc_my_rig.json`](../configs/experiments/mpc_my_rig.json) | `docs/results/sac_focused_policy.zip` | SAC swings up. MPC uses an eight-tick horizon and a 20 ms state prediction. |
| iLQR | [`ilqr_my_rig.json`](../configs/experiments/ilqr_my_rig.json) | `docs/results/sac_focused_policy.zip` | SAC swings up. iLQR uses an eight-tick horizon and a 12 ms state prediction. |

The classical presets predict the state under the previously sent command
before passing it to their respective balance controllers. This is a short
state prediction. The LQR preset uses its separate arm and pendulum delay
model. All four new presets passed a nominal 15-second simulation from hanging.
Their hardware results are pending.

In paired simulation comparisons on perturbed rig models, the classical
presets improved the final-ten-second pass counts and eliminated arm-guard
stops in the tested batches. Ten starts used the standard simulated feedback
and ten added feedback lag and a lower motor planner speed limit.

| Algorithm | Baseline | Transfer preset |
| --- | ---: | ---: |
| PID | 3/20 | 13/20 |
| MPC | 3/20 | 15/20 |
| iLQR | 3/20 | 14/20 |

The MPC preset completed all 643 optimization calls in its nominal smoke run.
The existing LQR fallback still occurred on some perturbed runs. For standalone
SAC, the limiter tied the raw policy at 32/48 passes under feedback stress and
reduced mean outward commands near the speed limit from 5.35 to 1.71 per trial.
Across all 72 paired SAC trials, the raw policy passed 51 and the limited policy
passed 50.

The later [`mpc_my_rig_coupling_delay.json`](../configs/experiments/mpc_my_rig_coupling_delay.json)
preset puts the 15 ms arm and 25 ms pendulum input delays inside MPC's horizon.
It uses the sent command history and checks arm position and speed before
capture. Against the single-lead MPC preset, it tied at 23/30 passes, reduced
balance releases from 22 to 2, and reduced optimizer fallbacks from 492 to 102
across the three paired simulation batches. Its nominal smoke run used MPC
for all 645 balance ticks. On hardware, runs `031705` and `031732` held for
2.54 s and 1.96 s respectively. After releases, the recovery policy commanded
up to 50 rad/s² while the pendulum was falling. Keep this preset for offline
testing until the loss-of-balance behavior has been addressed. For a graded
hardware run, use the [passing LQR setup](lqr-hardware-runbook.md).

For an attended hardware attempt, follow the [deployment procedure](deployment.md#deploy-a-student-controller)
with a current transfer preset and `docs/results/sac_focused_policy.zip`.

## Run now

From the repository root after installation:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pendulum_lab.cli sim --algorithm lqr --start upright \
  --episodes 10 --output runs/lqr_upright_01
PYTHONPATH=src python -m pendulum_lab.cli sim --algorithm lqr --start hanging \
  --episodes 1 --output runs/lqr_hanging_01
PYTHONPATH=src python -m pendulum_lab.cli train --algorithm ppo --start mixed \
  --config configs/ppo.json --output runs/ppo_01
PYTHONPATH=src python -m pendulum_lab.cli sim --algorithm ppo --start hanging \
  --checkpoint runs/ppo_01/policy.zip --episodes 20 --seed 10000 \
  --output runs/ppo_eval_01
```

Use a fresh output folder every time. Omitted `--params` means synthetic starter
parameters. After identification, add `--params configs/my_rig.json` to both
training and evaluation. Begin with upright balance, then hanging swing-up, then
multiple-seed full 15 s evaluation. The assignment requires the entire final 10 s
inside the angle and velocity gates with no safety stop.

## After assembly

Follow `docs/deployment.md`, collect identification data/video, export real rig
parameters, retune classical gains, and retrain/evaluate learned policies. Keep
hardware runs attended. Add real videos and paired simulator GIFs to `Report.md`;
never present synthetic parameters or pipeline smoke tests as measured results.

## Initial validation

See [starter baseline results](results/starter_baselines.json) for one complete
15 s hanging-start trial of each classical controller. The default energy gain,
arm centering gains, and capture gate were tuned on this synthetic model.
All four passed that deterministic trial, with 11.6 s continuous balance.
This initial classical check did not establish robustness or hardware transfer.
Subsequent learned-policy and stress-test results are recorded in [Report.md](../Report.md).

## GPU experiments and scaling

Set top-level `device` to `cuda` to request CUDA training. This requires a
CUDA-enabled PyTorch build and GPU access. For example, install a suitable
PyTorch CUDA wheel in the existing `.venv` following the official PyTorch
installation instructions. CUDA is an installation variant of the existing
PyTorch dependency; no new project dependency is required. Simulator physics
still runs on CPU. Small PPO MLPs can run faster on CPU because GPU transfers
and kernel launches outweigh the small network's computation.

Optional top-level `normalize_observations: true` applies fixed feature scaling
inside the saved network: each six-value frame is divided by
`[1.75, 1, 1, 5, 15, 1]`. This keeps angle, velocity, and action channels closer
in scale without changing the public observation interface. It is fixed scaling,
not running mean/variance normalization, and does not clip measurements. The
feature extractor and scaling buffer are saved in final and periodic policies,
so the normal controller loader works without a separate normalization file.
Old checkpoints retain their original preprocessing.

Optional `reward_scale` multiplies simulation training rewards by a finite
positive constant (default 1). Monitor logs show these scaled returns; compare
raw task metrics rather than rewards across differently scaled experiments.
This setting does not change the evaluation environment or success metric.
It can affect learning and SAC's balance between reward and entropy even when
the underlying task is unchanged.

The first CUDA batch is recorded in the ignored local folder
`runs/gpu_rl_20261016/`, with exact configs, source snapshots, dependency versions,
logs and intermediate policies. Do not claim success from a training curve;
evaluate saved policies in separate full-length hanging-start episodes.

For PPO with `device: cuda`, optional `cpu_rollouts: true` collects observations
and actions on CPU, moving the policy to CUDA between rollouts for batched
optimization. This avoids a device transfer for every small-network action.
It does not change the PPO objective or replace the learned controller. The
optimizer updates still run on GPU. Normal `Controller` loading remains usable
on CPU for evaluation and deployment. The hybrid path was checked with actual
CUDA training and a CPU checkpoint reload on the RTX 4070 Ti.

Reproducible starting configurations from this batch are `configs/ppo_gpu.json`,
`configs/sac_gpu.json`, and `configs/dqn_gpu.json`. For example:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pendulum-lab train --algorithm dqn \
  --start mixed --config configs/dqn_gpu.json --output runs/dqn_gpu_new
pendulum-lab sim --algorithm dqn --checkpoint runs/dqn_gpu_new/policy.zip \
  --output runs/dqn_gpu_new_eval
```

These are initial tested settings, not an optimized multi-seed benchmark.

## Continuing SAC

SAC accepts `resume_checkpoint` and optionally `resume_replay_buffer` in its
training config. `total_steps` then means **additional** transitions; the saved
step count is preserved. A resumed policy keeps its saved architecture,
preprocessing, optimizer and algorithm settings. The new environment, seed,
reward scale and checkpoint options apply to the continuation. Keep the reward
scale and observation contract consistent with the saved experience.

If no replay file is available, the weights/optimizer are restored but experience
is not. `resume_warmup_steps` (default 10000) collects fresh transitions before
critic updates. Set `save_replay_buffer: true` to save `replay_buffer.pkl` beside
the final policy and replay snapshots beside intermediate policies. These
buffers can be tens of MB; they remain ignored experiment artifacts.

## Robustness evaluation

The separate `pendulum_lab.robustness` command evaluates saved policies only in
simulation. It preserves the standard 15 s success criterion and safety envelope.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pendulum_lab.robustness \
  --policy ppo=runs/gpu_rl_20261016/ppo_hybrid_train/policy.zip \
  --policy dqn=runs/gpu_rl_20261016/dqn_train/policy.zip \
  --output runs/robustness_new --episodes 10 --seed 30000
```

Use `--case NAME` to select cases. Full output includes parameter JSONs, true-state
traces, per-episode metrics, safety stops, and CPU action-computation latency.
The timing measurement excludes simulation stepping and serial communication;
it does not certify hardware control-loop timing.

The initial suite varies friction by factors 0.5/1.5, pivot inertia by 0.8/1.2,
transport delay to 20/30 ms, motor acceleration/speed caps, and additional
Gaussian angle/velocity measurement noise. Angles are perturbed before sin/cos
encoding and their noisy history is maintained consistently across steps.
Physics truth and reward are unaffected by measurement noise. Mild noise has
angle standard deviation 0.003 rad and velocity standard deviation 0.1 rad/s;
strong noise uses 0.01 rad and 0.3 rad/s. A combined case uses several mismatches.
These ranges are chosen stress scenarios, not identified uncertainty bounds.

Noiseless exact hanging starts are deterministic, so each fixed case is evaluated
once. Varied starts and noise cases use 10 held-out seeds. Repeating the same
noiseless trial under ten seed labels would not create independent evidence.
Failures here inform calibration and future domain-randomized training; testing
alone does not make a policy robust or change its network weights.

For fresh-clone installation, network details, exact training/resume/evaluation
commands, all stress-case settings, and artifact boundaries, see the
[experiment reproduction guide](reproducing-experiments.md).
