# SAC balancing learned from reward: assisted-start curriculum

Status: implemented and checked in simulation; first hardware lift attempt stalled (b00), lift revised.
This is separate from P1 ([sac-hardware-pass.md](sac-hardware-pass.md)). P1 stays as it is.

## Goal and constraints

Build a second SAC policy whose balancing comes only from SAC's own reward on
the rig, not from imitation.

- **Starting point is V1** (`runs/sac_finetune/V1/policy.zip`, sha256 `d27a50c6…0a66`),
  with its own actor and critics. Its whole history is SAC reward training:
  `rig_rl/sac` → disturbance fine-tune → `sac_long` → V1 (600k simulator
  steps). V1's `real_fraction` is 0, so its real logs were used only to
  monitor TD error.
- **Not used:** P1's weights, behaviour cloning, LQR commands, and other
  controllers' rig logs. `rig_curriculum.py` refuses any checkpoint whose
  lineage includes `pretraining.json`.
- **Final evaluation:** starts hanging, uses SAC alone, deterministic, with no
  lift and no assistance.

## Placement, release and engagement

Nobody touches the mechanism while the motor is engaged. A hand release is not
practical: upright, this pendulum diverges with a time constant of about
1/12.3 s ≈ 80 ms, so it passes 30° roughly 0.25 s after release.

Instead, the "placement" is done by a scripted lift. Every episode works like this:

1. **Reset and start check.** The normal reset brings the rig to rest, and the
   runner's own hanging start check runs, unchanged.
2. **Lift.** `LiftController` (in `src/pendulum_lab/rig_reset.py`) pumps
   pendulum energy toward a target just below upright. Three features make it
   work on the rig:
   - **Phase lead:** the pump acts on the swing predicted 40 ms ahead.
   - **Adaptive strength:** the pump grows while the energy is stuck far below
     the target.
   - **Per-swing correction:** at each turning point near the top, the lift
     corrects its energy aim by how far short the swing turned around.

   The arm-centring term sits outside the pump's own limit. Every tick goes
   through the same guards, deadlines and arm limiter as a normal run. In
   simulation the arm stays within about 40°.

   The first rig attempt (`b00`) used a fixed pump capped at a command of 0.6.
   It stalled 75–80° from upright with only ±15° of arm motion: each swing
   added only as much energy as it lost. Raising the pendulum damping 3–4× in
   the simulator reproduces that stall. The current lift reaches the window in
   2–3 s at 1× and 3× damping, and still times out at 5×.
3. **Handover.** On the first tick inside the stage window (pendulum angle,
   pendulum speed, arm angle, arm speed), SAC takes over. Its observation
   history holds the real preceding frames.
4. **Lift failure.** If the window is not reached within 20 s, the attempt is
   saved as `lift_timeout` and the rig resets. Three timeouts in a row end the
   session.

Lift ticks are saved as `lift_samples` for inspection only. They never become
training data. The lift is a reset tool, like the damping reset; it is not a
teacher.

## Placement by LQR (current first stage)

The stage `lqr_balance` places the pendulum with the rig-proven LQR preset
(`configs/lqr_my_rig_coupling_delay.json`; its swing-up is
`docs/results/sac_focused_policy.zip`). LQR swings the pendulum up, catches
it and holds it. After LQR has held for 0.3–0.6 s and the state is inside the
window, SAC takes over. LQR's ticks are never training data: LQR decides only
where SAC starts, never what SAC does. Disclose this in the report.

The window is sized from LQR's real holds:

| Measure | 50th percentile | 90th percentile |
|---|---|---|
| Pendulum angle | 1.4° | 5.2° |
| Pendulum speed | 0.4 rad/s | 1.3 rad/s |
| Arm angle | 8° | 31° |
| Arm speed | 0.85 rad/s | 3.4 rad/s |

The resulting window is: pendulum under 10° and 2 rad/s, arm under 40° and
3 rad/s. The first training batch (`b03`) used holds of 0.8–1.5 s with the
arm under 30°. On that day LQR's holds were only 1.2–1.6 s, with the arm
around 30–40°, so three lifts in a row timed out. Replaying those recordings
against the new rule hands over in all five. LQR reached a 0.8 s hold in 15 of 17 rig runs, after 3–12 s.

**Yardstick:** LQR's own rig numbers, measured over the same 17 runs, are
what SAC has to approach.

| Measure | LQR median |
|---|---|
| Longest hold | 4.6 s (mean 5.2 s) |
| Holds over 1 s, per run | 1 |
| Arm drift while upright | 8.8 °/s |
| Limiter override | 13% of ticks |

## Curriculum and workflow

Stages are defined in `configs/rig_curriculum.json`:

| Stage | Handover window | Purpose |
|---|---|---|
| `balance` | under 12°, under 3 rad/s | learn to balance |
| `capture` | under 20°, under 4.5 rad/s | learn to catch |
| `recover` | under 30°, under 6 rad/s | catch from harder states, recover |
| `mixed` | `recover` window, plus about one third hanging starts | keep swing-up |
| `hanging` | no lift | graded evaluation |

`scripts/rig_curriculum.py` has three commands:

- **`collect`** runs one batch on one fixed checkpoint. Actions are stochastic
  for training, or `--deterministic` for evaluation. It saves every attempt:
  episodes, lift samples, handover state, policy transitions, and the
  checkpoint's sha256.
- **`update`** runs offline SAC updates (critic, actor and entropy terms) on
  the policy ticks of the chosen batches. It writes a new version directory
  with `policy.zip` and `version.json`, recording the parent, the batches used
  and their checkpoint hashes. Versions are never overwritten, so rolling back
  means using an older one.
- **`report`** prints, per batch: balance duration, holds over 1 s, arm drift
  while upright, limiter intervention, and full-task successes.

**Promotion gate:** a new version is used only if its deterministic hardware
evaluations improve those measures, not just reward or critic loss. Before
promotion, a version must also still pass the standard simulator check.

## Simulation check (rig-matched simulator)

Setup: `SimClient`, which includes 18 ms delay, arm lag and the stepper's
zero-crossing kick. Training was 4 rounds on the `balance` stage. Each round
had 10 stochastic episodes of 6 s, then 6000 updates at learning rate 3e-5;
the first round also had a 2000-step critic warmup.

| Version | Balance stage, deterministic: mean longest balance | Arm drift | Hanging, 20 s, SAC alone | Standard simulator |
|---|---|---|---|---|
| V1 | 0.91 s | 47 °/s | 0/4 (1.4 s) | 10/10 |
| v1 | 3.93 s | 20 °/s | – | **0/10** |
| v2 | 6.00 s (whole episode) | 6 °/s | 4/4 (18.5 s) | 10/10 |
| v4 | 6.00 s | 4 °/s | 4/4 (18.0 s) | 10/10 |

What this shows:

- The pipeline works end to end.
- V1 fails near upright in the rig-matched simulator, just as it does on the rig.
- Ten reward-only episodes were enough to change that in simulation.
- v1 briefly lost the standard simulator task, which is why every version
  must pass the gate before use.
- **Caveat:** this simulator matched past rig failures but did not predict new
  policies on the rig before, so the deciding evidence must come from hardware.

The pilot script and data are in `runs/curriculum_sim/` (gitignored).

## Hardware plan (one bounded step at a time)

1. **Lift check plus V1 baseline.** Run V1 deterministically on the `balance`
   stage, 4 episodes, no learning.
2. **Pilot training batch.** Run V1 stochastically on the `balance` stage for
   10 episodes, then update to v1 offline.
3. **Evaluate v1.** Run deterministic `balance` and deterministic `hanging`
   batches. Continue only if they improve.
4. **Continue the curriculum.** Move through `capture`, then `recover`, then
   `mixed`, under the same gate.
5. **Final run.** Use the standard hardware runner from hanging, with SAC
   alone, and record video.

## Hardware log

| Batch | Checkpoint | Stage | Result |
|---|---|---|---|
| `b00` | V1, deterministic | `balance` (pump lift) | 3/3 lifts stalled 75–80° from upright; lift redesigned, then replaced by LQR placement |
| `b02` | V1, deterministic | `lqr_balance` | 3 handovers; longest holds 0.70 / 0.16 / 1.42 s (baseline) |
| `b03`, `b04` | V1, stochastic | `lqr_balance` | 11 training episodes, 4,378 transitions |
| v1 | V1 + 8,756 updates on `b03` + `b04` | – | rig-matched sim: 9.2 s hold from hanging, 2/4 passes; standard sim 6/10 |
| `e01` | v1, deterministic | `lqr_balance` | mean longest hold 0.75 s, against V1's 0.76 s: no change |
| `e02` | v1, deterministic | `hanging`, 20 s | longest holds 0.04–0.3 s, spinning 45–61% of the time, against V1's 0.2–0.9 s and 34–40% (`reset_check_v1`): worse |

**v1 is rejected; back to V1.** The simulator predicted a large gain from
this much data, but the rig showed none.

Only about 0.8 s of each 8 s episode was near upright, so the batches held
roughly 400 balancing ticks. From now on, `lqr_balance` episodes end 0.5 s
after the pendulum first passes 45° (`end_after_fall_s`). That stops time
being spent on spinning, and collects more balance attempts per minute.
| `b05` | V1, stochastic | `lqr_balance`, ending on fall | 10 episodes (6 lift timeouts: LQR's holds were short, with the arm about 40° off centre); in total, all batches hold 29 s of near-upright rig data |
| v2 | V1 + 9,646 updates on rig data only | – | standard sim 0/10: rejected without a rig test |
| v2m2 | V1 + 60k SAC steps, each batch 50% V1's training simulator and 50% rig data | – | standard sim 10/10; on rig near-upright states its actions differ from V1's (mean difference 0.39, correlation 0.66) |

Training on rig data alone overwrites everything the rig data does not cover;
v1 lost its swing-up on the rig. `update --sim-steps` instead mixes the rig
data with fresh experience from V1's own training simulator (RLPD-style, the
same method V1 itself was trained with), so the rest of the policy stays
anchored. It is still pure SAC reward learning.
| v2_u250 / u1000 / u2500 / u5000 | V1 + 2000 critic-only steps + N updates on the same rig data | – | standard sim 0/10 for all four, even 250 updates; zero updates and critic-only both give 10/10, so the update tool is sound |

Decision rule (user, 2026-11-07): evaluate v2 on hardware first. If it fails,
try fewer updates on the same dataset. V1 stays unchanged as the baseline,
and deterministic hardware results decide which checkpoint is kept.

Where the rig-only versions fail in the standard simulator (8 starts from
hanging, 15 s):

| Version | Reaches upright | Longest hold | Time spinning |
|---|---|---|---|
| V1 | 8/8 | 13 s | 3% |
| v2_u250 | 8/8 | 0.8 s | 17% |
| v2 | 8/8 | 0.1 s | 47% |

Their swing-up is intact; what changed is their balancing, the part the rig
data trains. Simulator balancing is exactly what has not predicted the rig:
V1 holds 13 s in simulation but 0.2–0.9 s on the rig. So for these versions
the simulator score is not evidence either way, and the hardware decides.

### 2026-11-08: encoder fixed, v2 and v2m2 on the rig

`e03`, `e03b` and `e03c` (v2) were stopped by encoder faults (bits=2) before any
episode finished. After the encoder repair, no faults occurred in 8 episodes.

| Batch | Checkpoint | Stage | Result |
|---|---|---|---|
| `e04` | v2, deterministic | `lqr_balance`, 6 s | mean longest hold 0.44 s, 0 holds over 1 s: worse than V1 (0.75 s) |
| `e05` | **v2m2**, deterministic | `lqr_balance`, 6 s | **held the whole 6 s in 4/4 episodes**; pendulum within 1.7–8.8° of upright, arm drift 2.1 °/s, limiter 0.2% |

The e05 readings are live, not a frozen encoder. The pendulum angle changes by
0.14–0.31° per tick. The longest run of identical readings is 8 ticks. Repeated
readings make up 18–26% of ticks, compared with 16% in LQR's holds. The arm moves
across 20–100° per episode.

The 6 s episode limit caps the hold, so v2m2's real hold time is not yet known.
For comparison, V1 holds 0.75 s and LQR's median longest rig hold is 4.6 s.
v2m2 is pure SAC reward learning: V1, plus SAC updates on a 50/50 mix of V1's
simulator and the rig ticks from its own episodes. It passes the standard
simulator 10/10. Next: a deterministic `hanging` evaluation with SAC alone, then
the official runner.

| Batch | Checkpoint | Stage | Result |
|---|---|---|---|
| `e06` | **v2m2**, deterministic | `hanging`, 20 s, SAC alone | longest holds **11.68 / 1.58 / 13.32 / 3.40 s** (mean 7.49 s), against 0.08–0.3 s for v1 (`e02`) and 0.2–0.9 s for V1; two of four end upright |

**Runner checkpoint.** `versions/v2m2/policy.zip` stores its replay-buffer class
as `finetune_real.MixedReplayBuffer`, so `pendulum-lab hardware` cannot load it
(`ModuleNotFoundError`). `versions/v2m2_runner/policy.zip` has the same weights
(every tensor checked equal), with the buffer class set to SB3's `ReplayBuffer`.
It loads without `scripts/` on the path.

**Simulator disagreement.** In the standard simulator runner (`pendulum-lab sim`,
`my_rig.json` + `sac_my_rig.json`), v2m2 fails on every seed (0.32 s hold), and
so does the original file; V1 passes (13.18 s). The "standard sim 10/10" recorded
for v2m2 above was not reproduced by this command. Here the simulator and the rig
disagree in the opposite direction from before: the rig passes and the simulator
fails.

### Official runner, v2m2_runner: 0/10 passes (2026-11-08 03:18–03:24)

Longest holds: 6.4, 0.5, 0.1, 7.8, 0.5, 1.1, 1.9, 3.2, 1.1 and 0.4 s. Run `032011`
was still balancing when the run ended. The rig was healthy throughout: 20.0 ms
ticks with none late, no encoder jumps, and the same start state as `e06`. One
earlier attempt (`031340`) stopped on encoder fault bits=2.

**Why: v2m2 balances, but it rarely catches.** Its swing-up reaches upright by
about 2.2 s, but too fast: the pendulum is moving at 4–10 rad/s at 20°, often
with the arm already 30–65° off centre. Most passes are held for only 0.1–0.3 s.
A long hold starts only when one pass happens to arrive slowly enough. That came
at 4.4–7.1 s in the official runs and at 5.0–5.8 s in `e06`. A 15 s pass needs a
catch before about 5 s.

This matches the training data. Every rig episode v2m2 learned from began inside
LQR's balance window (under 10°, under 2 rad/s), so the rig taught it to balance
but never to catch. Its catching still comes from V1's simulator training.
Next: collect rig data from v2m2's own swing-up arrivals.

### Catch batch and v3

| Batch / version | Checkpoint | Stage | Result |
|---|---|---|---|
| `b06` | v2m2, stochastic | `hanging`, 15 s, SAC alone | 10/10 episodes, no faults; 7,480 transitions with 151 arrivals near upright; mean longest hold 1.63 s |
| v3 | v2m2 + 60k SAC steps, each batch 50% V1's training simulator and 50% rig data (`b03`–`b05`, `e05`, `e06`, `b06`; 17,487 transitions) | – | standard simulator runner: **pass** (11.68 s; v2m2 fails, 0.32 s). Agreement with v2m2 on rig states: balancing (`e05`) correlation 0.88, sign 87%; swing-up and arrival (`b06`) correlation 0.76, sign 79% |

v3 changed mainly where it was meant to (arrivals), and changed less on the
balancing it already did well. Runner copy: `versions/v3_runner/policy.zip`, with
identical weights. Gate: v3 replaces v2m2 only if it catches earlier and passes
more often on the rig, from hanging.

| Batch | Checkpoint | Stage | Result |
|---|---|---|---|
| `e06_..._2` | v2m2, deterministic | `hanging`, 20 s | 4 further episodes, reaches the top by 2.2 s every time (holds in the summary) |
| `e07` | v3, deterministic | `hanging`, 15 s | **the swing-up broke**: swings stall 115–126° from hanging for seconds, and it first gets near upright at 3.5–12.5 s (v2m2: 2.2 s every time); once up it held 5.2 s and 2.5 s; mean longest hold 2.05 s |

**v3 rejected.** Most of `b06` is swing-up and spinning, so the rig data also
rewrote the swing-up. On the rig the swing-up got weaker, even though in the
simulator it passed. `update --rig-max-theta-deg` now keeps only rig ticks whose
latest frame is within that angle of upright. **v3b** = v2m2 + the same update,
using rig ticks within 45° (10,581 of 17,487), so the rig shapes catching and
balancing, and the swing-up is left to the simulator half, as it was in v2m2.

| Batch / version | Checkpoint | Stage | Result |
|---|---|---|---|
| `e08` | **v3b**, deterministic | `hanging`, 15 s, SAC alone | **swing-up intact** (top by 2.2–2.7 s); longest holds **6.1 / 9.0 / 0.4 / 8.6 s** (mean 6.0 s vs v2m2's ~2.3 s in 10 official runs); in two episodes the hold lasted to the end of the run (caught at 6.0 and 6.4 s, final 10 s 91% / 87% balanced); one early catch at 2.7 s held 6.2 s |
| v3c | v2m2 + 20k steps at lr 3e-5, same data and 45° filter | – | standard sim pass (12.8 s); closer to v2m2 than v3b (swing-up corr 0.93 vs 0.82); not yet on the rig |

v3b passes the gate: it beats v2m2 on 15 s holds from hanging, and its swing-up
survived. Runner copy: `versions/v3b_runner/policy.zip`.

**Official runner, v3b_runner:** 0/3; longest holds 0.6, 1.3 and 2.9 s. This is
not a runner difference: rebuilding the observations from the logged readings
reproduces the logged commands exactly, both for the runner and for
`rig_curriculum.py`. It is variance. Pooled from hanging:

| Policy | Episodes | Holds ≥ 6 s | Holds ≥ 10 s | Passes |
|---|---|---|---|---|
| v2m2 (`e06`, `e06_..._2`, 10 official) | 18 | 4 | 2 | 0 |
| v3b (`e08`, 3 official) | 7 | 3 | 0 | 0 |

Each arrival near upright becomes a long hold only about 5–10% of the time, and
it arrives at 3–10 rad/s. Further rig rounds of the same kind give small,
uncertain gains.

### v4: penalize fast arrivals (started 2026-11-08)

The failure is arrival speed (3–10 rad/s near upright). V1's reward already has
an overspeed penalty near upright (`k_upright_overspeed`), but its weight of 0.02
is negligible: about 0.7 per tick at 8 rad/s, against about 25 per tick for
being upright. `update --overspeed-weight 0.3` raises it in both halves: in the
simulator's reward, and in the stored rig rewards, recomputed from `next_obs`.
At 8 rad/s that is about 11 per tick. It is still reward-only SAC.
v4 = v3b + 100k steps, 50/50, rig ticks within 45° (15,321 of 27,463, now
including `e06_..._2`, `e07` and `e08`), overspeed weight 0.3.

**v4 rejected (no rig test).** With the overspeed weight at 0.3, SAC stopped swinging up entirely:
in the rig-matched simulator (6/6 episodes) and the standard simulator, it parks
the arm at the travel limit and the pendulum stays below 27°. The training return
fell from 2558 to 2313. The penalty made approaching upright costly, so not
swinging up became the better option. Next: v4b, a milder change (weight 0.08,
40k steps, lr 3e-5), with `b07` added.

| Batch / version | Checkpoint | Stage | Result |
|---|---|---|---|
| `b07` | v3b, stochastic | `hanging`, 15 s | mean longest hold **4.45 s**, against **1.63 s** for v2m2 under the same conditions (`b06`); holds ≥ 6 s: 2/10 (0/10 for v2m2); episode 6 caught at 2.7 s and held 10.3 s |
| v4b | v3b + 40k steps at lr 3e-5, overspeed weight 0.08, rig ticks within 45° (19,349) | – | **rejected without a rig test**: arrivals not slower in the rig-matched simulator (median 8.8 rad/s vs v3b's 7.9), and it reaches the top later (3.7 s vs 2.1–3.2 s); passes the standard simulator (11.0 s) |

The like-for-like comparison (`b06` vs `b07`, both stochastic) confirms that
v3b catches better than v2m2. Shaping the reward for slower arrivals did not
work: at 0.3, SAC stopped swinging up; at 0.08, arrivals did not slow down.
**v3b is the final reward-only SAC policy.**

Unusable runs (no samples, cancelled, interrupted, or cut short by a sensor
fault) were moved to `runs/archive/`, listed in `runs/archive/README.md`.

**Official runner, second session (03:55–03:58).** v3b_runner, 4 runs: longest holds 6.58
(caught at 8.2 s, still balancing at the end), 1.10, 1.50 and 0.52 s. v3c_runner, 1 run: 0.40 s.
In total, v3b has made 7 official attempts with 0 passes. Across all 11 deterministic v3b
episodes from hanging, there were 4 holds of 6 s or more, but only one catch started
before 5 s (`e08` episode 0, 2.7 s, held 6.1 s).

## Why v3b's catches end (v3b frozen; analysis after Codex's review)

Data: every deterministic v3b episode from hanging (`e08`, `e08_..._2`, and 7
official runs), with `b07` added for the entry-state comparison. A "catch" is
a balanced stretch of at least 0.5 s, using the runner's own test (|θ| ≤ 15°,
|θ̇| ≤ 4 rad/s). Script: `scripts/catch_endings.py`.

**How the 18 catches that ended early were lost:**

| Ending | Catches | Early catches (started before 5 s) |
|---|---|---|
| Pendulum lost with the arm free (arm inside ±55°, limiter idle, motor command saturated on 0–20% of ticks) | **14** | **6 of 7** |
| Arm at the travel limit, or limiter active in the last 0.5 s | 4 | 1 of 7 |
| (Held to the end of the run) | 4 | – |

Most losses are therefore not arm drift and not limiter intervention. The
pendulum is lost while the arm still has room and the motor has headroom. The
θ error does not grow slowly; the catch fails quickly (median 0.57 s,
interquartile range 0.38–0.97 s).

**Arrival state does not predict the outcome** (medians at the first balanced tick):

| | n | θ | θ̇ | Peak θ̇ in the 0.2 s before | Arm | Arm speed | Mean \|action\|, first 0.5 s | Mean \|action\|, next 0.5 s | Arm speed, first 0.5 s |
|---|---|---|---|---|---|---|---|---|---|
| Long holds (≥ 4 s) | 9 | 11.0° | 3.3 | 21.6 | 36° | 1.3 | 0.41 | 0.22 | 1.3 |
| Lost catches (0.3–4 s) | 36 | 9.1° | 3.0 | 19.8 | 32° | 1.3 | 0.47 | 0.30 | 1.8 |

Long holds and lost catches start from the same states, including equally fast
arrivals. So "arriving too fast" is **not** what separates them, and the v4/v4b
idea was aimed at the wrong thing. The difference shows up only after the
catch: holds that survive calm down to small commands (about 0.13 once settled),
while lost catches keep commanding larger actions with a faster arm, and leave
the window within about half a second. With 9 against 36 cases, these medians
are indicative, not conclusive.

**Implication for the next training change.** The weak part is the catch
transient: the first ~1 s after entering the window, from states the policy
sometimes handles and sometimes does not. In the rig data, these ticks are a
small share of the near-upright set, which is dominated by settled balancing.
A targeted change is to oversample rig ticks from the first second after each
window entry in the rig half of the update, and to collect more of them
(stochastic hanging episodes with v3b). Not done yet.

### v5: catch-transient oversampling

`update --catch-oversample K` marks rig ticks from 0.2 s before to 1 s after each
entry into the balance window (`catch_mask`, computed from the newest obs frame
of each episode) and repeats them K times in the rig half.
v5 = v3b + 40k steps at lr 1e-4, 50/50, rig ticks within 45° (20,579; 10,916 of
them catch ticks, ×2, about 70% of the rig half), all batches through `b07`.

Offline: swing-up intact in the rig-matched simulator (top at 2.2–2.7 s, against
v3b's 2.1–3.2 s); passes the standard simulator (11.34 s). Agreement with v3b on
v3b's own rig states: catch transient corr 0.80, settled balance 0.88, swing-up
0.77. Its actions are not smaller (|a| 0.52 vs 0.50 in the catch transient), so
whether its catches hold up better can only be tested on the rig.
Runner copy: `versions/v5_runner`.

| Batch | Checkpoint | Stage | Result |
|---|---|---|---|
| `b08` | v3b, stochastic | `hanging`, 15 s | mean longest hold 2.33 s (against 4.45 s in `b07`, same policy); 1 hold ≥ 6 s |
| `e09` | **v5**, deterministic | `hanging`, 15 s | **swing-up broke on the rig**: swings stall 125–140° from hanging, with the arm pinned near −78° (range −79…+32°); 0.00 s balance in 4/4. Both simulators had shown a normal swing-up (top at 2.2–2.7 s) |

**v5 rejected.** It is the second time (after v3) that an update produced a
swing-up that looked fine in simulation but stalled on the rig, both times with
the arm working against one side of its travel. A rig-matched simulator swing-up
is therefore not a sufficient gate: every candidate needs a short rig check of
the swing-up.

**No rig drift.** The hanging reading at the start of every episode from 02:58 to
04:08 stays within ±1° (most within ±0.5°), and loop timing is unchanged. The
batch-to-batch spread is variance in the policy's outcomes, not the rig changing.

**v3b, all hardware episodes from hanging (15 s):**

| | Episodes | Mean longest hold | Median | ≥ 6 s | ≥ 10 s | Passes |
|---|---|---|---|---|---|---|
| Deterministic (`e08`, `e08_..._2`, 7 official) | 15 | 2.90 s | 1.50 s | 4 | 0 | 0 |
| Stochastic (`b07`, `b08`) | 20 | 3.39 s | 3.01 s | 2 | 1 | 0 |

`e08` (mean 6.0 s) was the best batch, not a typical one.

## What went wrong (analysis, 2026-11-08 ~04:20)

### What each version was trained on

All versions are SAC reward updates. The simulator half is always V1's training
simulator (fixed pendulum friction from `my_rig.json`, with no friction
randomization). The rig half is SAC's own logged transitions, with rewards
recomputed by the same reward function.

| Version | Parent | Simulator steps, lr | Rig data used (real ticks) | Rig result |
|---|---|---|---|---|
| v2m2 | V1 | 60k, 1e-4 | `b03`–`b05` (V1 stochastic, LQR-placed, near upright): 4,823 | swing-up ✅; balance from placement 6 s 4/4 |
| v3 | v2m2 | 60k, 1e-4 | + `e05`, `e06`, `b06` (**including swing-up and spinning ticks**): 17,487 | **swing-up broke** |
| v3b | v2m2 | 60k, 1e-4 | same batches, **only ticks within 45° of upright**: 10,581 | swing-up ✅; mean hold 2.9 s (det.) |
| v5 | v3b | 40k, 1e-4 | everything through `b07`, within 45°, catch ticks ×2: 31,495 | **swing-up broke** |

### Findings

1. **The rig-tested broken swing-ups stall at the last pump.** v2m2, v3b, v3 and
   v5 all reach 118–137° from hanging within 2 s. The working ones then reach the
   top by 2.5–3 s; v3 and v5 stay at 125–140°.
2. **Every update changes the whole policy by a similar amount.** On the same
   real swing-up states, v3, v3b and v5 each differ from their parent with
   correlation 0.7–0.9 in every angle band (0–180° from upright). The amount of
   change does not separate broken from working; the rig outcome of the swing-up
   after an update is effectively a draw. Two of the four updates tested on the
   rig broke it.
3. **The simulators did not catch either break.** Both showed normal swing-ups
   for v3 and v5.
4. **Friction explains part of it.** `scripts/swingup_damping_check.py` scales
   the pendulum's friction in the rig-matched simulator. At 3×, v3 stalls at
   126–129° (on the rig, 115–126°) while v2m2 still reaches the top in 3 of 4
   episodes, so v3's pump has the least margin. But v5 swings up at 2× and stalls
   higher (150°) at 3×, and v3b, which works on the rig, also stalls at 3×. So
   friction is not the whole difference, and a high-friction simulator would not
   have been a reliable gate.
5. **The catch transient is the near-upright weakness** (previous section):
   catches are lost within ~0.6 s with the arm free, from entry states that
   match those of long holds.

### Mistakes in this session

- The "arriving too fast" hypothesis was wrong (finding 5), so v4 and v4b
  targeted the wrong thing.
- `e08` (v3b's best batch) was presented as typical; the pooled mean is 2.9 s, not 6 s.
- Simulator swing-up checks were treated as a sufficient gate; v5 passed them
  and failed on the rig.

### Implication

The near-upright work can only pay off if updates stop moving the swing-up.
The swing-up must be held fixed during training (keep the actor's outputs away
from upright close to the parent's, which is rig-proven), not re-learned each time.

### v6 / v6b: swing-up anchor

`update --anchor-min-deg D --anchor-steps K`: after every SAC training call, the
actor takes K extra steps per gradient step that pull its deterministic action on
real rig states at least D from upright back to the parent's (v3b, whose rig
swing-up works). It is a trust region on SAC's own previous policy. The anchor
states come from every hanging batch, including v3's and v5's broken swings.
Both versions: v3b + 40k steps at lr 1e-4, 50/50, rig ticks within 45° with catch
ticks ×2 (37,044 rig transitions, data through `e09`).

Agreement with v3b on v3b's real rig states (correlation):

| From upright | v5 (no anchor) | v6 (anchor ≥ 45°, ×1) | v6b (anchor ≥ 30°, ×2) |
|---|---|---|---|
| 0–15° | 0.86 | 0.80 | 0.79 |
| 15–30° | – | 0.53 | 0.69 |
| 30–45° | 0.71 (15–45°) | 0.80 | 0.95 |
| 45–90° | 0.70 | 0.97 | 0.99 |
| 90–180° | 0.82 | 0.99 | 0.99 |

| Simulator check | v3b | v6 | v6b |
|---|---|---|---|
| Rig-matched, friction ×1: reaches the top at | 3.0–3.2 s | 2.6–3.1 s | 2.1 s |
| Rig-matched, friction ×2: reaches the top | 4/4 | 1/4 | 3/4 |
| Standard simulator | pass | fail (0.56 s) | pass (12.8 s) |

**v6b goes to the rig; v6 is held back.** v6's 30–45° band (part of the final
pump) moved, and its swing-up margin dropped.

| Batch | Checkpoint | Stage | Result |
|---|---|---|---|
| `e10` | **v6b**, deterministic | `hanging`, 15 s, SAC alone | **first full-task pass by reward-only SAC on the rig (episode 0)**: caught at 3.0 s and held 12.0 s to the end; final 10 s 100% balanced; the runner's success test passes. Swing-up intact: top at 2.2–2.3 s in 4/4 (no stall). The other episodes: holds 1.4, 0.6/0.5, 2.0 s. Mean longest hold 4.0 s (v3b: 2.9 s) |

This pass comes from `rig_curriculum.py collect` (same observations, limiter and
success test as the runner; checked earlier by replaying logged observations).
It is not yet an official `pendulum-lab hardware` run. **No video was recorded of this pass.**

**Official runner, v6b_runner (04:20–04:21):** 0/3; longest holds 4.70 s (caught at 10.3 s, held to the end), 0.72 s and 0.58 s.
Swing-up intact in all three (top at 2.2–2.9 s).
(The first two attempts could not load the checkpoint: the anchor's training wrapper had been saved
into it. Fixed in `update_mixed` and in `v6b_runner`; weights unchanged.)

**v6b vs v3b, all deterministic episodes from hanging (15 s):**

| | Episodes | Mean longest hold | Median | Passes | Catches ≥ 0.3 s per episode | Early catches (< 5 s) held ≥ 2 s | Median lost-catch length |
|---|---|---|---|---|---|---|---|
| v3b | 15 | 2.90 s | 1.50 s | 0 | 1.9 | 2 of 10 | 0.64 s |
| v6b | 7 | 3.14 s | 1.38 s | 1 | 2.1 | 1 of 8 | 0.48 s |

The `e10` pass was an outlier. **The anchor worked** (swing-up intact in 7/7, unlike
v3 and v5), but **oversampling the catch ticks did not improve the catch**: v6b's
catch numbers match v3b's.

### Why the catch training did not take: the simulator half dominates near upright

With v3b acting stochastically in V1's training simulator (the simulator half of
every update), **74% of simulator transitions are within 45° of upright and 63%
within 15°** (the simulator resets 25% of episodes upright, and v3b balances well
there). With a 50/50 mix, about 43% of all near-upright training samples came
from the simulator, whose near-upright physics never matched the rig. The rig's
catch data was competing with that.

Now that the anchor holds the swing-up, the simulator is only needed for swing-up.
`update --sim-min-deg D` draws the simulator share of each batch only from
transitions at least D from upright (`swingup_sim_buffer`), so near upright the
policy learns from rig transitions alone. Checkpoints saved by `update_mixed` now
carry SB3's plain `ReplayBuffer` class and no anchor wrapper, so the runner loads
them directly. **v7** = v6b's recipe (anchor ≥ 30° ×2, catch ×2, rig within 45°)
+ `--sim-min-deg 45`, data through `e10`.

v7 offline: the near-upright actions changed much more than v6b's (corr with v3b
0–15° 0.66, 15–30° 0.47; v6b 0.79, 0.69), so the rig-only near-upright training
took effect. Swing-up held (30–45° 0.91, ≥45° 0.97–0.99); normal friction: top at
3.0 s 4/4; standard simulator pass (10.5 s). **But at 2× friction it reaches the top
0/4** (v3b 4/4, v6b 3/4). The swing-ups that broke on the rig scored ≤ 2/4 there
(v3 1/4, v5 2/4), so v7's swing-up margin is a risk. v7b (anchor ×4, simulator
share from ≥ 30°) is training as a backup; v7 goes to a 4-episode rig check now.

| Batch | Checkpoint | Stage | Result |
|---|---|---|---|
| `e11` | v7, deterministic | `hanging`, 15 s | **swing-up intact on the rig** (top at 2.3–2.8 s, 4/4), so the 2× friction check was a false alarm here; holds 1.56, **9.22 (caught at 5.8 s, held to the end, final 10 s 95%)**, 0.34, 0.36 s; 0 passes; mean longest 2.87 s; only 3 catches ≥ 0.5 s in 4 episodes |

v7 is not better at catching than v3b or v6b (4 episodes; a small sample).

**Why offline updates keep missing the catch (assessment).** Every update since v3
learns near upright from a fixed pile of rig data collected by *other* policies
(V1, v2m2, v3b, v5...). SAC's critic can only judge actions that resemble those
in the data; the new policy's own catch behaviour is never in the data until it
has been run. That is the standard offline-RL limitation: big jumps from old
data, then a surprise on the rig. The remedy is a proper on-rig loop with small
steps: collect stochastic episodes with the *current* policy, update from it
(anchor on, lr low), repeat, and evaluate deterministically every round.

## Catch direction and mirrored rig data (2026-11-08, offline)

### The policies catch in one direction only

Linear fit of the action to (θ, arm, pendulum speed, arm speed) over the first
0.6 s of each catch (deterministic hanging episodes):

| | θ gain, catches held ≥ 2 s | θ gain, catches lost | R², lost |
|---|---|---|---|
| LQR (rig) | −4.8 | | |
| v3b | −3.2 | +0.4 | 0.15 |
| v6b | −2.4 | −0.6 | 0.14 |
| v7 | −7.2 | −1.6 | 0.32 |

In held catches SAC acts like LQR. In lost catches it is not stabilizing at all.
Splitting catches (≥ 0.3 s in the balance window) by the arm's direction at entry:

| | Arm moving − at entry: held ≥ 2 s / lost | Arm moving +: held / lost |
|---|---|---|
| v3b, v6b, v7 (`e08`–`e11`) | **6 / 6** | **1 / 16** |
| all deterministic SAC hanging runs (incl. official runner) | 14 / 37 (27%) | 7 / 45 (13%) |
| LQR, all rig runs | 24 / 28 (46%) | 40 / 56 (42%) |

LQR catches equally well both ways, and in steady balance its mean θ is about
+0.2° with symmetric actions, so **the rig is symmetric; the policies are not.**
Feeding each policy the mirrored observation (arm, sin θ, arm speed, pendulum
speed and last action negated) and negating its output should give the same
action for a symmetric controller (correlation 1.0). On real rig states:

| From upright | V1 | v2m2 | v3b | v6b | v7 | **v8** |
|---|---|---|---|---|---|---|
| 0–15° | 0.42 | 0.57 | 0.65 | 0.72 | 0.69 | **0.78** |
| 15–45° | 0.41 | 0.57 | 0.64 | 0.61 | 0.32 | **0.57** |

(Swing-up, ≥ 45°, is asymmetric in every version; that is harmless and is held
by the anchor.) The asymmetry is inherited from V1 and was never trained out,
because the rig data is mostly catches from one side.

### `update --mirror` and v8

The rig and the reward are mirror-symmetric (every reward term is squared or an
absolute value), so every real transition also holds mirrored, with the same
reward. `--mirror` adds the mirrored copy of every kept rig transition. This is
SAC's own data, so the update stays reward-only.

**v8** = v7's recipe exactly (parent v3b, 40k simulator steps, lr 1e-4, seed 3,
50/50, anchor ≥ 30° ×2, rig ≤ 45°, catch ×2, `--sim-min-deg 45`), data through
`e11`, plus `--mirror` (83,264 rig transitions). The only differences from v7 are
the mirror and `e11`.

| Offline check | v7 | v8 |
|---|---|---|
| Mirror symmetry 0–15° / 15–45° | 0.69 / 0.32 | 0.78 / 0.57 |
| Agreement with v3b ≥ 45° (swing-up) | 0.95–0.98 | 0.93–0.98 |
| Rig-matched sim, friction ×1: top at | 3.0 s 4/4 | 3.0 s 4/4 |
| Rig-matched sim, friction ×2: reaches the top | 0/4 | 3/4 |
| Standard simulator (`pendulum-lab sim`, hanging, 20 s) | pass, 15.5 s | pass, 17.8 s |
| Loads in the official runner without `scripts/` | yes | yes |

`report` now also prints, per batch, early catches (start < 5 s) held ≥ 2 s and
held/lost catches by arm direction at entry (`catch_stats`).

### Rig test plan (Codex's suggestion plus the direction metric)

Parent and candidate interleaved, so rig drift hits both equally. Recipe frozen.
Five rounds of 4 deterministic hanging episodes each, v8 then v7 → 20 episodes
per policy (about 40 catches each; 4-episode batches cannot separate versions).
Decision fixed in advance: v8 replaces v7 if its swing-up stays intact and its
pooled held-catch share and arm-moving-+ held share are both higher; otherwise
the mirror is dropped. If v8 wins, the next step is the small-step loop:
stochastic batch with v8 → modest `update` (same recipe, `--mirror`) → interleaved
deterministic comparison.

## v7 on the rig: balancing into the arm guard, and v9 (arm-guard reward)

`e12_v7_hanging_det_r1b` (4 episodes, 20:48): 9 of 12 catches ended with the arm at
−77° to −80° (the 1.35 rad travel limit) and the limiter overriding 50–88% of
commands in the last 0.5 s; every one drifted negative at 10–40°/s, and several
began at −60° to −68° because the swing-up leaves the arm off-centre. Mean θ near
upright was −0.15°, so this is not an encoder offset. `e12_v8_hanging_det_r1`
(mirrored data): swing-up intact, but 0 of 9 catches held 2 s (mean 0.98 s).

Fitted arm-centering feedback in settled balance (action per 10° of arm offset):
LQR −0.049 (all rig runs), v3b −0.007, v6b −0.016, v7 −0.016 (`e11`) / **+0.011**
(`r1b`). The policies barely centre the arm. The reward's centring term
(`k_motor_pos` 0.5 · arm², 0.9 at the guard) is negligible against the ~25 per
tick of upright and stillness bonuses.

**Reward change (allowed: the assignment fixes interface, metrics and safeguards,
not the training reward).** `RewardWeights.k_arm_guard` (default 0, so nothing
else changes) adds `k · exp(−θ²/(30°)²) · max(|arm| − 45°, 0)²`: zero below 45°
of arm, 6.2 per tick at 77° near upright, almost zero during swing-up.
`update --guard-weight K --guard-start-deg D` applies it to the recomputed rig
rewards and to the simulator half.

**v9** = v7 + 20k simulator steps, lr 1e-4, seed 3, 50/50, anchor ≥ 30° ×2 (to
v7), rig ≤ 45°, catch ×2, `--sim-min-deg 45`, `--guard-weight 20`, data through
`e12` (46,847 rig transitions; 10,688 rig ticks penalized). No mirror.

| Offline check | v7 | v9 |
|---|---|---|
| Agreement with v7 ≥ 30° (swing-up) | – | 0.95–0.99 |
| Arm-centering gain on real near-upright states (per 10°) | +0.015 | −0.022 (LQR's sign) |
| Mean action with the arm shifted to −60° / +60° | +0.08 / −0.01 | +0.21 / −0.03 |
| θ gain on the same states | −2.15 | −1.20 (weaker) |
| Rig-matched sim swing-up, friction ×1 / ×2 | 4/4 / 0/4 | 4/4 / 4/4 |
| Rig-matched sim, hanging 15 s | 0/4, 0.7 s | **3/4 passes, 11.8 s**, arm within ±40° |
| Standard simulator | pass | **fail** (catches, oscillates, arm to 77°) |

v9 breaks this doc's standard-simulator gate. It goes to the rig anyway, because
the two simulators disagree in opposite directions (v7 fails the rig-matched one
yet held 9.2 s on the rig) and neither has predicted rig balancing.
Next: v9 and v7 alternating on the rig (deterministic, hanging), then a
stochastic v9 batch as fresh training data for the next small update.

| Batch | Checkpoint | Result (deterministic, hanging, 15 s, 3 alternating rounds × 4) |
|---|---|---|
| `e13_v9_hanging_det_r1`–`r3` | v9 | mean longest hold 1.82 s, 0/12 passes, early catches held ≥ 2 s 1/7; **16 of 27 catches still end at the guard**, now on both sides (drift 20–50°/s) |
| `e13_v7_hanging_det_r1`–`r3` | v7 | mean 1.80 s, 0/12 passes, early ≥ 2 s 1/8; 9 of 26 end at the guard, 10 with the limiter ≥ 40% |
| `b09_v9_hanging_train` | v9, stochastic | 10 episodes, mean 0.87 s (training data) |

**v9 did not fix the guard problem on the rig.** Its offline centering probe changed,
but on the rig the arm still runs into the guard during a hold, with the pendulum leaning the opposite way
(arm → −77°, pendulum ends +16° to +20°, and vice versa): the policy holds the
pendulum slightly tilted and keeps accelerating the arm under it. Stopping that
needs the non-minimum-phase move LQR makes (tilt the pendulum back first); the
offline reward change did not produce it. v7's own mean fell from 2.87 s (`e11`)
to 1.80 s, so 4-episode batches were indeed misleading.
