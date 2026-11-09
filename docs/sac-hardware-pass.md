# SAC on the real rig: how it passed, and what it took

Written for whoever updates [`Report.md`](../Report.md). Everything here is
standalone; the last section has text ready to paste into §1, §7.6, §9 and §11.

## Result

**SAC passed a full 15 s hardware run on its own:**
[`docs/results/hardware/sac/20261107_215833`](results/hardware/sac/20261107_215833/),
11.68 s continuous balance, 79% of the run balanced, final 10 s inside the
success window. Command: the normal runner, deterministic SAC, the usual
SAC preset with its arm limiter.

```bash
.venv/bin/pendulum-lab hardware --algorithm sac --params configs/my_rig.json \
  --config configs/sac_my_rig.json \
  --checkpoint docs/results/rig_rl_disturbance/sac_lqr_pretrained/policy.zip \
  --port /dev/ttyUSB0 --duration 15 --output runs/sac_<timestamp>
```

(The run itself loaded the same file from `runs/sac_finetune/P1/policy.zip`;
the promoted copy is byte-identical, sha256 `396595023ece…10929`.)

- Checkpoint: [`rig_rl_disturbance/sac_lqr_pretrained/policy.zip`](results/rig_rl_disturbance/sac_lqr_pretrained/)
  with its lineage ([`pretraining.json`](results/rig_rl_disturbance/sac_lqr_pretrained/pretraining.json),
  [`v1_training.json`](results/rig_rl_disturbance/sac_lqr_pretrained/v1_training.json)).
- Simulator GIF (standard simulator, seed 2026, 13.2 s balance, passes 10/10 seeds):
  [`sac_lqr_pretrained.gif`](results/sac_lqr_pretrained.gif).
- Real video: **none recorded for the passing run** (TODO: film further runs).

**Pass rate.** About one in three, similar to LQR (4 of 18). Evidence, all
deterministic unless stated:

| Where | Passes | Notes |
| --- | --- | --- |
| Official runner | 1/1 | the run above |
| Rig session, deterministic | 2/6 | [`rig_sac/p1_deterministic_vs_sampled`](results/rig_sac/p1_deterministic_vs_sampled/): each pass held 13.2 s |
| Rig session, sampled actions | 0/6 | same session, alternating; sampling does not help |
| Rig session, first try | 0/3 | [`rig_sac/p1_first_evaluation`](results/rig_sac/p1_first_evaluation/): best 2.4 s |

Runs are all-or-nothing: P1 either keeps the first catch for the rest of the
run or loses it within a second and does not recover (the arm then swings to
the limiter and the pendulum spins). First-arrival speed and arm position do
not predict which.

## How the passing policy was made (three stages)

1. **SAC in simulation** — [`sac_long`](results/rig_rl_disturbance/sac_long/), the team's
   existing rig-tuned policy. On hardware: 0/7, best 1.50 s.
2. **V1: SAC fine-tuned in simulation with noisy velocity readings** (per-episode
   pendulum/arm velocity noise 0.4–1.2 rad/s), 600k steps, no hardware data:
   ```bash
   .venv/bin/python scripts/finetune_real.py --checkpoint docs/results/rig_rl_disturbance/sac_long/policy.zip \
     --output runs/sac_finetune/V1 --steps 600000 --real-fraction 0 --velocity-noise 0.4 1.2 --seed 1301
   ```
   On hardware: 0/4, best 2.14 s.
3. **P1: V1's actor pretrained on LQR's real balancing** — the step that worked:
   ```bash
   PYTHONPATH=src:scripts .venv/bin/python scripts/pretrain_actor.py \
     --checkpoint runs/sac_finetune/V1/policy.zip --output runs/sac_finetune/P1 --steps 4000
   ```
   - Demonstrations: the 18 LQR hardware runs with the working preset
     (`lqr_my_rig_coupling_delay.json`); only ticks in balance mode, inside holds
     of at least 1 s, minus the last 0.3 s before a loss: 4,531 observations, each
     with LQR's own command. Every fifth run (4 runs, 683 ticks) is held out.
   - Anchor: 3,420 simulator states away from upright (|θ| > 0.5 rad) with V1's
     own actions, so the SAC-learned swing-up is kept.
   - Loss: mean-squared error of the actor's mean action; only the actor changes.
   - Held-out agreement with LQR: correlation 0.76 → **0.989**, sign 67% → **96%**,
     mean error 0.28 → 0.03; swing-up anchor error 0.004.

At run time P1 is an ordinary SAC policy in `algorithms/sac.py`: deterministic,
no LQR, no capture gate, the same arm limiter as every SAC run.

**Why it worked when nothing else did.** Every earlier attempt relied on the
simulator being right near upright, and it never was (see below). LQR is the
only controller proven on this rig, and its preset was hand-tuned on the
hardware (e.g. 15 ms arm vs 25 ms pendulum response). Imitating its commands on
real states moves that rig-specific knowledge into the network without having
to model the rig.

## What we learned on the way (sim-to-real for §9)

All SAC hardware attempts (15 s, deterministic):

| Policy | What changed | Runs | Passes | Best hold |
| --- | --- | ---: | ---: | ---: |
| `sac_long` | baseline | 7 | 0 | 1.50 s |
| A / D | + delay randomization, real-state resets | 3 / 3 | 0 | 1.80 / 0.52 s |
| F | + command-rate penalty | 3 | 0 | 1.20 s |
| J | + learned dynamics correction (exploited by the policy) | 1 | 0 | — |
| R50 | + half hardware data in replay (RLPD-style) | 3 | 0 | 1.22 s |
| V1 | + velocity-reading noise | 4 | 0 | 2.14 s |
| W1 / W2 | + identified arm lag, stepper kick, reading jitter | 3 / 5* | 0 | 0.70 / 1.90 s |
| **P1** | **+ actor pretrained on LQR's rig balancing** | **1 + 9** | **1 + 2** | **13.2 s** |

\* one W2 start was rejected by the start check (not a policy result).

Findings, each measured, with the experiment that tested it:

1. **The velocity reading has no extra lag** beyond the firmware's 8 ms
   backward window (fitted extra lag 0.5 ms with device timestamps).
2. **Rig readings jitter far more than the simulator's:** pendulum angle
   0.6° rms from smooth motion vs 0.045°, velocity 0.6 rad/s vs 0.1. The jitter
   grows with command size and is heavy-tailed. A model fitted on older runs
   predicted held-out runs' jitter within 20%, but adding it to the simulator did
   not reproduce the failures (everything still passed). An earlier 0.8 rad/s
   white-noise model did reproduce them, but it was stronger than what was
   measured: a symptom match, not a cause.
3. **The stepper does not stop where commanded.** Balanced ± acceleration
   blocks leave 0.2–0.9 rad/s of residual arm speed, because FastAccelStepper
   jumps through zero speed (~0.2–0.3 rad/s per zero crossing). A kick model
   fitted on one capture cut held-out arm-prediction error 23° → 10° and
   7° → 2.5°. Captures: `sysid accel --waveform reversal` (new, below).
4. **With LQR's own capture, SAC still cannot balance:** a diagnostic
   (`custom.py`, LQR swing-up and capture, then SAC balancing) held 0/47
   captures for 2 s vs LQR's 15/49. So the problem was balancing, not arrival.
   On the very states LQR handled, SAC's commands agreed with LQR's in sign only
   56% of the time (85% in simulation).
5. **The simulator is too optimistic near upright for every controller:**
   started from the same 96 real capture states, the simulator holds 95–98% for
   LQR and SAC alike; the rig gave 43% (LQR) and 5% (SAC).
6. **The pendulum responds about one tick later than the step count**
   (closed-loop identification with a known dither on LQR's command:
   lag centroid 2.06 ticks vs the simulator's 1.6–1.7). A simulator with
   10–20 ms of physical arm lag reproduced this and the outcome pattern (LQR
   holds some captures, SAC none), yet policies trained in it (W1, W2) still
   failed on the rig. Three validated simulator changes in a row failed to
   predict a new policy, which is why we stopped relying on the simulator.
7. **On-rig SAC learning, as tried, made things worse.** A 20-episode pilot
   from P1 (automatic resets, mixed replay, LQR imitation term, learning
   rate 3e-5) degraded the deterministic evaluations from P1's level to
   1.1 → 2.0 → 0.8 → 0.3 s. The passing policy has no on-rig training.

## Code added or changed (for "submit any code modifications")

Defaults are unchanged everywhere, so the classical controllers and their
results are unaffected; all 97 tests pass.

| File | What |
| --- | --- |
| `src/pendulum_lab/plant/pendulum_env.py` | Optional simulator knobs, all off by default: pendulum-reading lag, coupling gain, physical arm lag behind the step count (`_arm_lag_s`). |
| `src/pendulum_lab/plant/actuator_transport.py` | Optional stepper zero-crossing kick (`zero_cross_kick_rad_s`, default 0); test in `tests/test_actuator_transport.py`. |
| `src/pendulum_lab/sysid_accel.py` | New `--waveform reversal` (tick-to-tick ± reversals, nominally returns to rest); test in `tests/test_sysid_driven.py`; noted in `docs/system-identification.md`. |
| `src/pendulum_lab/algorithms/custom.py` | Diagnostic options: SAC balancing behind LQR's capture (`balance_policy`), and a known dither on LQR's balance command for identification (`balance_dither`). Default behaviour unchanged. |
| `src/pendulum_lab/rig_reset.py`, `rig_session.py`, `scripts/rig_sac.py` | Attended on-rig sessions: automatic reset between episodes, same guards as `hardware.run`, optional SAC learner; `tests/test_rig_session.py`. |
| `scripts/finetune_real.py` | SAC fine-tuning (mixed replay, velocity noise, command jitter, arm lag, kick, tilt). V1 came from here. |
| `scripts/pretrain_actor.py` | LQR-guided actor pretraining. P1 came from here. |
| `scripts/train_rig_rl.py`, `scripts/real_state_starts.py`, `src/pendulum_lab/residual.py` | Earlier experiments (delay/physics randomization, real-state resets, learned correction). |

Analysis scripts are in `runs/obs_model/` (ignored by git).

## Rules check

- **"PPO, SAC and DQN policies swing up and balance on their own."** Met at run
  time: P1 runs through the standard SAC controller with no classical code.
  *But* its balancing was learned by imitating LQR's logged commands
  (behaviour cloning), and its swing-up by SAC. The assignment does not forbid
  demonstrations, but it is not pure RL. **Say so plainly in §7.6**; if in
  doubt, ask the TA before the demo.
- **Residual RL / hybrid control:** not used in P1. The LQR+SAC handoff exists
  only as a diagnostic in `custom.py` and none of its runs are reported as SAC.
- **"Keep each required algorithm's implementation and helpers in its provided
  file" and "RL training uses `train(...)`".** The SAC deployment code is
  `algorithms/sac.py`, but P1's training pipeline lives in `scripts/`
  (as does the team's earlier rig fine-tuning, `train_rig_rl.py`). This
  departs from the letter of the rule; either document it or move the
  pipeline into `sac.train` before submission.
- **"Store the lab contract, plant identity … with the checkpoint."** P1 ships
  with lineage files but no embedded contract, same as `sac_long`.
- **Hardware safeguards:** unchanged. Firmware untouched; every tick of the
  on-rig sessions used the runner's guards and deadlines; the motor was never
  re-engaged after a stop; reset moves use the firmware's bounded position
  command. Every session was attended.
- **Same physical task:** simulator changes are opt-in only; the standard
  simulator and evaluation metrics are unchanged.

## Ready to paste into Report.md

**§1 summary table, SAC row:**

```markdown
| Learning | SAC | 1 / 8 official (+2 / 6 in rig sessions) | 11.68 s | 13.2 s in rig session | 10 / 10 (standard sim) |
```

**§7.6 SAC (replace "Hardware so far"):**

```markdown
**Hardware: passed.** [`20261107_215833`](docs/results/hardware/sac/20261107_215833/): full 15 s,
11.68 s continuous balance. Pass rate about one in three (2 of 6 further deterministic
episodes in a rig session, [data](docs/results/rig_sac/p1_deterministic_vs_sampled/)).

Checkpoint: [`sac_lqr_pretrained/policy.zip`](docs/results/rig_rl_disturbance/sac_lqr_pretrained/).
Simulator GIF: [`sac_lqr_pretrained.gif`](docs/results/sac_lqr_pretrained.gif).

**How it was trained.** (1) SAC in simulation (`sac_long`); (2) fine-tuned in simulation with
noisy velocity readings (V1); (3) the actor pretrained to reproduce LQR's commands on LQR's
real balancing states (4,531 rig observations, 4 runs held out), anchored to V1's own actions
away from upright to keep the learned swing-up. Held-out agreement with LQR rose from
correlation 0.76 to 0.99. At run time the policy acts alone (deterministic SAC, standard arm
limiter); **its balancing was learned from LQR demonstrations, not from reward alone.**

Every simulator-only variant (delay and physics randomization, real-state resets, smoothness
penalty, learned dynamics correction, mixed real/sim replay, measured noise, arm lag and
stepper kick) failed on hardware (0/31, best 2.14 s); see
[docs/sac-hardware-pass.md](docs/sac-hardware-pass.md).
```

**§9 sim-to-real (add):** findings 1–7 above, in short. The key lines:

```markdown
- The simulator is too optimistic near upright: from the same 96 real capture states it holds
  95–98% for LQR and SAC, the rig 43% (LQR) and 5% (SAC).
- The stepper does not stop where commanded (residual 0.2–0.9 rad/s after balanced reversals;
  FastAccelStepper jumps through zero speed), and the pendulum responds about one tick later
  than the step count. Both were measured and modelled, but simulators with them still did not
  predict new policies on the rig.
- What transferred was the rig-tuned behaviour itself: SAC passed only after its actor imitated
  LQR's real balancing.
```

**§11 AI usage (rows):**

```markdown
| 2026-11-07 | Claude | "i want to make this work … make sure you are not overfitting" | Sim-to-real diagnosis: velocity-reading analysis, stepper reversal captures, SAC-behind-LQR diagnostic, dithered closed-loop identification | Each simulator fix was validated on held-out data; three that matched past failures still failed on new policies, and were reported as such |
| 2026-11-07 | Claude, Codex, Sol (review) | "check sol advice and if it's good" | On-rig session tool with automatic resets; LQR-guided actor pretraining | The pretrained SAC passed on hardware; on-rig learning degraded it and was stopped |
```

## Still open

- **Video** of a passing SAC run (none recorded for 20261107_215833).
- More official runs to pin down the pass rate.
- Whether to move the training pipeline into `algorithms/sac.py` (rules check above).
- Nothing in this work is committed yet.
