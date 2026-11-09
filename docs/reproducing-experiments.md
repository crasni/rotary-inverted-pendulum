# Reproduce the simulation experiments

This guide covers the implemented code and experiments through 2026-10-16.
See [algorithm derivations and interfaces](algorithms.md), the
[results and saved checkpoint index](../Report.md), and
[hardware commissioning](deployment.md). All reported controller results use
synthetic starter parameters. Physical identification, hardware trials, videos,
and the team's own reflection are still pending.

## Environment

The tested environment was an isolated repository `.venv`, Python 3.12.15,
PyTorch 2.14.1+cu130, and an NVIDIA RTX 4070 Ti. Python itself was not compiled.
[environment.txt](../environment.txt) records installed package versions;
[pyproject.toml](../pyproject.toml) remains the dependency declaration. No new
project dependency was added. The virtual environment is not committed.

For a fresh clone, from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl]'
```

Evaluation can run on CPU. For the CUDA installation variant used in these
experiments, with a compatible NVIDIA driver:

```bash
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`device: cuda` requests GPU training; it does not fall back silently to CPU.
The simulator runs on CPU. PPO's tested GPU config uses `cpu_rollouts: true`,
so action collection is on CPU and gradient updates are on CUDA. SAC/DQN use
CUDA directly. Our desktop execution sandbox required GPU access outside the
sandbox; if this environment reports CUDA unavailable, check host access before
assuming the GPU or driver is missing.

## Evaluate the committed models

Use a new output directory on every invocation. These commands require no robot
and default to CPU inference and the synthetic starter parameter file.

```bash
pendulum-lab sim --algorithm ppo \
  --checkpoint runs/gpu_rl_20261016/ppo_hybrid_train/policy.zip \
  --output runs/recheck_ppo
pendulum-lab sim --algorithm dqn \
  --checkpoint runs/gpu_rl_20261016/dqn_train/policy.zip \
  --output runs/recheck_dqn
pendulum-lab sim --algorithm sac \
  --checkpoint runs/sac_robustness_20261016/train/policy.zip \
  --output runs/recheck_sac
```

The default episode is 15 s from exact hanging rest. DQN requires the committed
`policy.zip.actions.json` beside its ZIP. Input scaling is embedded in the
trained networks; do not normalize observations a second time externally.
Use `--start upright --episodes 5 --seed 10000` for balance diagnostics.
Exact hanging resets are deterministic; repeating them with different seed
labels does not create varied initial states. For GIF export, add `--gif`;
headless rendering in this environment uses `MUJOCO_GL=egl`.

## What networks and settings were trained?

The 100,352-step CPU PPO pilot used separate 64/64-neuron Tanh actor and critic
networks with raw observations, and failed hanging-start evaluation. The later
GPU experiments changed scaling, network size, batch settings, exploration,
and training duration together. Their improvement does not isolate which change
caused success; this was manual experimentation, not a controlled hyperparameter
search or a multi-training-seed benchmark.

All later models receive 24 values: four frames of arm angle, sin/cos pendulum
angle, arm velocity, pendulum velocity, and previous action. Each frame is divided
inside the network by `[1.75, 1, 1, 5, 15, 1]`.

| Algorithm | Saved network structure | Main training settings |
| --- | --- | --- |
| PPO | Separate actor and value MLPs: 24 → 128 → 128 → 1, Tanh hidden activations; learned Gaussian action standard deviation | 8192-step rollouts, batch 1024, 10 epochs, entropy bonus 0.01, reward scale 0.05 |
| SAC | Actor: 24 → 128 → 128 with mean/log-standard-deviation heads; two critics: 25 (state plus action) → 128 → 128 → 1; ReLU hidden activations, target critics | Batch 256, automatic entropy coefficient, replay capacity 200000, one gradient update per four transitions, reward scale 0.2 |
| DQN | Q-network: 24 → 128 → 128 → 5, ReLU hidden activations; target network | Batch 128, replay capacity 200000, exploration anneals from 1 to 0.05 over the first 30% of training, reward scale 0.1 |

All use learning rate 0.0003, discount 0.99, and training seed 0. PPO/SAC output
continuous normalized acceleration; DQN selects from `[-1, -0.5, 0, 0.5, 1]`.
Training samples exploration actions; the controller loaders evaluate
deterministically. There is no PID/swing-up helper attached to a learned policy.
Deployed policy weights stay fixed; observation feedback changes actions, not
weights. [Checkpoint-inspected network structures and settings](results/trained_networks.json)
include defaults not explicitly written in the configs. SAC's saved
`learning_starts: 310000` is the absolute step threshold after its 300000-step
checkpoint plus 10000 fresh replay-warmup transitions.

## Train and continue

The initial GPU training commands used mixed hanging/upright resets and:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pendulum-lab train --algorithm ppo \
  --start mixed --config configs/ppo_gpu.json --output runs/ppo_fresh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pendulum-lab train --algorithm sac \
  --start mixed --config configs/sac_gpu.json --output runs/sac_fresh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pendulum-lab train --algorithm dqn \
  --start mixed --config configs/dqn_gpu.json --output runs/dqn_fresh
```

PPO completes whole rollouts: its requested million steps produced 1,007,616.
SAC/DQN each trained 300000 steps. The initial fully-CUDA PPO attempt was stopped
because action collection was slow, then restarted with CPU rollouts and CUDA
updates. Reported PPO runtime excludes that prototype. Jobs overlapped, so their
runtimes are not an isolated GPU-versus-CPU speed comparison.

To repeat the recorded SAC continuation using the committed 300000-step model:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pendulum-lab train --algorithm sac \
  --start mixed --config configs/sac_continue_gpu.json \
  --output runs/sac_continuation_new
```

This adds 300000 transitions, reaching 600000 total. It restores weights and
optimizer settings, collects 10000 fresh transitions because the old replay
buffer was not saved, and saves new replay buffers. A resumed policy retains
its saved architecture/preprocessing/algorithm settings; changing the `sac`
constructor object in the resume config does not retune an already loaded model.

To continue the latest model, copy that config to a new file and change
`resume_checkpoint` to `runs/sac_robustness_20261016/train/policy.zip`.
Set `total_steps` to the desired **additional** budget. If the local
`runs/sac_robustness_20261016/train/replay_buffer.pkl` exists, also set
`resume_replay_buffer` to that path. Replay is not in Git, so a fresh clone must
omit that key and collect fresh warmup experience. Keep `reward_scale: 0.2` and
the same observation contract when restoring replay. Resumption without replay
is a warm start, not an exact restoration of the original training trajectory.

Training outputs are `policy.zip`, periodic models under `checkpoints/`,
`learning_curve.monitor.csv`, `training.json`, and the CLI's `run.json` manifest.
SAC additionally records `continuation.json` and, when requested, replay files.
Architecture/scaling are stored in each model; DQN additionally needs its grid
sidecar. Saved experience should only be restored with matching preprocessing,
reward units and environment semantics.

## Repeat robustness validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pendulum_lab.robustness \
  --policy ppo=runs/gpu_rl_20261016/ppo_hybrid_train/policy.zip \
  --policy dqn=runs/gpu_rl_20261016/dqn_train/policy.zip \
  --policy sac=runs/sac_robustness_20261016/train/policy.zip \
  --episodes 10 --seed 30000 --output runs/robustness_recheck
```

Each policy is evaluated with frozen weights on the same cases. The stress suite
uses the original safety envelope and truth-based final-10-second success metric.
The physics changes affect the simulated plant, not the trained network. Gaussian
noise affects observations and their history, not truth-based rewards/metrics.

| Case name | Change from the starter model |
| --- | --- |
| `nominal_exact` | Exact centered, still, hanging reset |
| `varied_starts` | Arm within ±5°, pendulum within ±0.05 rad of hanging, zero initial velocities |
| `friction_low`, `friction_high` | Both viscous and Coulomb friction multiplied by 0.5 or 1.5 |
| `inertia_low`, `inertia_high` | Pivot inertia multiplied by 0.8 or 1.2; mass/COM unchanged |
| `delay_20ms`, `delay_30ms` | Command transport delay 20 or 30 ms, versus nominal 10 ms |
| `noise_mild` | Independent angle noise std 0.003 rad; velocity noise std 0.1 rad/s |
| `noise_high` | Independent angle noise std 0.01 rad; velocity noise std 0.3 rad/s |
| `motor_weaker` | Motor acceleration cap 40 rad/s²; planner speed cap 4 rad/s |
| `combined` | Varied starts; friction ×1.5, inertia ×1.1, delay 20 ms, mild noise, acceleration cap 45 rad/s², speed cap 4.5 rad/s |

Noiseless fixed-start cases run once; stochastic cases run ten trials. These
stress bounds are hypotheses, not measured hardware confidence intervals. Inertia
changes must still imply physically positive COM inertia. The current bounds are
valid for the supplied starter file; check validity for an identified rig.

For the recorded six-column comparison, run the same suite separately on the
committed SAC 300000/350000/450000/600000-step models. Each invocation accepts at
most one checkpoint per algorithm. The JSON at
[results/robustness.json](results/robustness.json) contains all comparison metrics,
case settings, model hashes and the continued SAC holdout result.

To repeat the final SAC's extra 20 varied-start trials:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pendulum_lab.robustness \
  --policy sac=runs/sac_robustness_20261016/train/policy.zip \
  --case varied_starts --episodes 20 --seed 40000 \
  --output runs/sac_holdout_recheck
```

Outputs include per-case parameter JSONs, episode traces and metrics, plus a
summary with safety stops and CPU action latency. Latency excludes serial IO,
physics stepping and whole-loop scheduling. Case comparisons helped compare
checkpoints; they are validation evidence, not hardware-success probabilities.
No domain-randomized training was performed in the 2026-10-16 experiments.
The 2026-10-22 randomized SAC experiment is documented below.

## Artifact boundaries and validation

The [checkpoint table](../Report.md#saved-checkpoints) lists the six ZIPs and DQN
sidecar explicitly committed despite ignore rules. Selected GIFs, traces, JSON
metrics, figures, configs, network descriptions and
[episode-return data](results/learning_curves.csv) are committed too. The CSV
records original reward units after undoing training-only reward scaling; its
phase column distinguishes initial training from SAC continuation.

Most `runs/` files remain local: raw training logs, full robustness episode
traces/parameter files, replay buffers, source snapshots, discarded/intermediate
models not in the checkpoint table, and temporary experiment scripts. Detailed
JSON may reference those historical paths; that is not a promise that every
intermediate model is included in Git. Committed summaries/selected traces and
the evaluation commands above permit review and new evaluations of saved models.
Training again can give different weights/results across device/library versions.

The final code validation used:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/rpl-mpl OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests
python -m build --no-isolation
```

58 tests passed and source/wheel builds completed. Coverage includes classical
contracts and a full LQR trial, short RL train/save/load checks with scaling,
SAC continuation with/without replay, noise history/seeding, unchanged physical
truth under sensor noise, parameter-file isolation, and existing infrastructure
and firmware-safety regressions. Short training tests are not performance tests.

Before hardware conclusions, complete commissioning and system identification,
replay real commands against the simulator, measure full-loop timing, retrain or
retune for the identified rig, and collect attended trials/videos. Further
simulation work is training across measured parameter/noise/delay uncertainty
and repeating training with multiple seeds. Team members must add their own
verification/reflection to the report.

## Randomized SAC experiment (2026-10-22)

Run from the repository root with the existing RL environment and CUDA:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python -u -m pendulum_lab.randomized_training \
  --output runs/sac_randomized_20261022 --steps 600000
```

Use a fresh output path when repeating. This warm-starts the existing 600000-step
SAC policy, discards nominal replay, gathers 10000 fresh warmup transitions, and
adds 600000 transitions with CUDA updates. Saved architecture and optimizer
settings come from the resumed policy. Each episode independently samples uniform
friction factors 0.5–1.5, pivot-inertia factors 0.8–1.2, delay 10–30 ms,
acceleration limits 40–50 rad/s², speed limits 4–5 rad/s, angle-noise standard
deviation 0–0.01 rad, and velocity-noise standard deviation 0–0.3 rad/s.
These are synthetic hypotheses, not measured rig bounds. Mixed upright/hanging
starts and the existing hardware safety envelope remain in use. Noise changes
observations, while rewards and evaluation metrics use physical truth.

The experiment saves every sampled episode's settings in `physics.jsonl` and
its GPU, contract, ranges, seed and configuration in `experiment.json`. It
validates the baseline and six periodic checkpoints plus the final policy on
eight combinations with three episodes each. The best newly trained checkpoint
is selected by successes, then fewer safety stops, then mean continuous balance.
The selected checkpoint and baseline are tested on twenty independent combined
settings with ten episodes each, plus nominal exact and the old combined case.
Exact noiseless nominal starts run once. Final test seeds/settings are separate
from validation and do not select checkpoints. Summaries include checkpoint
hashes, case settings, truth-based success, safety stops and arm travel; traces
are retained locally. One training seed cannot establish training reproducibility.
The completion marker is `complete.json`; absent that file, the pipeline has not
finished. Outputs remain ignored under `runs/` until selected artifacts are
explicitly preserved for the report.

## Failure-focused SAC continuation (2026-10-22)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python -u -m pendulum_lab.focused_training \
  --output runs/sac_focused_20261022
```

Use a new output directory for a repeat. Configuration is in
`configs/sac_focused_gpu.json`. This experiment resumes the preserved randomized
SAC checkpoint at 1,200,000 steps and adds 300,000 CUDA training transitions
(seed 174), collecting 10,000 fresh warmup transitions without restoring replay.
It uses the nine combinations on which that policy previously failed at least
one trial. Anchors are weighted by failure count: 25% of episodes use an exact
anchor, 40% perturb an anchor with Gaussian standard deviation equal to 8% of
each parameter's original range (clipped to that range), and 35% independently
sample the full original uniform ranges. Reset probabilities are 20% upright,
10% exact hanging and 70% perturbed hanging. Noise, reward, physics truth and
safety-envelope semantics are unchanged. Replay buffers are saved locally for
future continuation; they are not portable inference artifacts.

Previous test settings now inform training and no longer qualify as held out.
Twelve new validation combinations (six broad, six near failure anchors), five
trials each, compare the baseline, three periodic checkpoints and final policy.
Selection prioritizes fewer safety stops, then more successes, then longer mean
balance. The best newly trained candidate is tested; the baseline remains
eligible for the validation recommendation. In the recorded run the new final
policy won validation, but regressed on the fresh test, so the previous policy
remains the stronger general candidate. The chosen model is frozen before a fresh test
of 30 broad uniformly sampled combinations with ten trials each. Test seed
281000 differs from validation seed 181000 and training seed 174. These final
test results do not influence selection. Diagnostics revisit the nine failure
anchors with new episode/noise seeds (381000), nominal physics and the original
combined case. These diagnostics concern training physics, not generalization.

`experiment.json` records the GPU, parameter contract, source-report/checkpoint
hashes, sampling distribution and seeds. `physics.jsonl` records each sampled
case, source and anchor. `selection.json` records candidate scores;
`test_baseline/summary.json` and `test_focused/summary.json` record the fresh
comparison, and `diagnostics_*` hold the anchor checks. `complete.json` confirms
the whole pipeline finished. After completion, preserve artifacts with:

```bash
MPLCONFIGDIR=/tmp/rpl-mpl python scripts/summarize_focused_sac.py \
  runs/sac_focused_20261022
```

This exports a distinct policy, JSON comparison and plot under `docs/results/`,
leaving the previous experiment intact. All ranges remain synthetic and only one
continuation seed is trained. No hardware-success probability is established.
