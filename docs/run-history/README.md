# Project history: RPL2026 HW1, Group08 (rotary inverted pendulum)

Written 2026-11-10, after the report's final commits (11-09) and the scheduled live-demo day (11-09). Its purpose is to let anyone, including future me, understand **the whole project**: what the task was, what we built, what failed and why, what finally worked, the choices made along the way, and how the work with the AI assistants went. It was written just before the large training artifacts in `runs/` were deleted.

Sources: `Report.md`, `docs/*.md`, `logging.md`, `git log`, the metadata of every run in `runs/`, and the Claude Code session transcripts. Statements not backed by those sources are marked *(unclear)*.

**What else is in this folder**

| File | What it is |
|---|---|
| `README.md` | This file: the story, decisions, lessons and glossary |
| [`runs_index.md`](runs_index.md) / [`runs_index.csv`](runs_index.csv) | One row for each of the 543 run directories that were in `runs/`: date, algorithm, config, policy loaded, parent version, training method, episodes, passes, longest balance |
| [`metadata/`](metadata/) | Verbatim copy of every small file from `runs/`: `run.json`, `summary.json`, `batch.json`, `session.json`, `version.json`, `training.json`, training `.log`s, CSVs and scratch `.py` (1,423 files) |
| [`file_manifest.csv`](file_manifest.csv) | Every file that existed in `runs/` (3,863), with size and mtime, and sha256 for every deleted file and every model |
| [`kept_referenced.txt`](kept_referenced.txt) | Checkpoints exempted from deletion because a doc or test uses them |
| `../../scripts/inventory_runs.py` | The script that generated the index, manifest and `metadata/` |

---

## 1. The project in one page

**Task** ([assignment](../assignment.md)). Build a rotary (Furuta) inverted pendulum and identify its parameters. Then make **seven controllers** swing it up from hanging and balance it on the real rig: PID, LQR, iLQR, MPC (classical) and PPO, SAC, DQN (learning-based). **Pass** = run 15 s from a hanging start with no safety stop, and keep |θ| ≤ 15° and |θ̇| ≤ 4 rad/s for the whole final 10 s. Classical controllers may use any swing-up. Learned ones must swing up and balance by themselves. Every attempt counts.

**Rig.** Arduino Nano with a DRV8825 stepper driving the arm and an AS5600 magnetic encoder on the pendulum, 12 V supply. The host runs the `pendulum-lab` CLI at 50 Hz over USB serial at 500 kbaud. Commands are normalized arm acceleration (1 = 50 rad/s²). Firmware guards trip at ±100° arm angle and 6 rad/s arm speed.

**People.** crasni (me) built and ran the rig and drove most of the November work. Sean Cheng did the simulator baselines, GPU RL training and the `rl/rig-training` branch. `uuuu`/`uuuu9923` did the robustness evaluation, the LQR delay model and most of `Report.md`. `takatakata195` made one commit. AI tools used: **Codex** (first LQR hardware tuning; later critiques), **Claude Code** (most of the November code, analysis and the report audit), and **"Sol"**, a reviewer whose messages were pasted into sessions *(unclear whether a teammate or another AI)*.

**Final result: all seven controllers passed on the real pendulum.**

| Method | Passes / attempts | Report run | Longest balance | Final version | Date of pass |
|---|---:|---|---:|---|---|
| PID | 1 / 8 | `20261107_161650` | 10.42 s | `pid_my_rig_lqr_match.json` | 11-07 |
| LQR | 4 / 18 | `20261107_030202` | 12.84 s | `lqr_my_rig_coupling_delay.json` | 11-07 |
| iLQR | 1 / 10 | `20261107_161026` | 13.26 s | `ilqr_my_rig_lqr_match.json` | 11-07 |
| MPC | 1 / 28 | `20261107_155613` | 10.66 s | `mpc_my_rig_transfer.json` (r = 30) | 11-07 |
| PPO | 3 / 17 | `20261109_013508` | 13.14 s | `ppo_ecG` | 11-09 |
| SAC | 2 / 12 reward-trained | `20261109_004713` | 13.12 s | `rnA` | 11-09 |
| DQN | 2 / 4 | `20261109_010341` | 13.56 s | `dqn_rn05` | 11-09 |

The classical controllers share one SAC swing-up plus a capture-gate hand-off. Only the balance law differs. Pass rates are low everywhere: roughly 1 in 5 for the classical controllers, and the learned ones passed only after the rig was repaired. The honest summary is that **every method can do it, and none is reliable**.

**Three lessons above all others:**
1. **Check the mechanics before blaming the model.** A pendulum slowly coming loose on its shaft made *every* controller fail for hours (and degraded runs for days). We spent that time on sim-to-real theories.
2. **Timing and the catch matter more than the control law.** Modelling the 15 ms arm and 25 ms pendulum delays and limiting arm speed at capture turned LQR from 0 passes into a working controller. PID, iLQR and MPC then passed by copying LQR's hand-off.
3. **Learned policies only transferred once the rig's real imperfections were in training**, in particular heavy-tailed sensor noise. Simulator "fixes" that each matched past failures (delay, arm lag, stepper kick, residual dynamics) did not predict new policies' rig results.

---

## 2. Timeline

| Dates | Phase | Outcome |
|---|---|---|
| 10-10 → 10-17 | Repo setup. Simulator baselines for all 7 controllers, GPU RL training (PPO/SAC/DQN), robustness sweeps, LQR delay model | All pass in the *starter* simulator. No rig yet |
| 10-22 | Randomized and failure-focused SAC (`sac_randomized_20261022`, `sac_focused_20261022`) | `sac_focused_policy.zip`, later the classical controllers' swing-up |
| 10-22 → 10-29 | Wiring bring-up | Reversed 12 V polarity, AS5600 I2C lock-ups, split motor coils, barrel-jack confusion. Final fix *(unclear, off-transcript)* |
| 11-06 evening | System identification (`sysid_*`, `accel_*`, `envelope_*`), `configs/my_rig.json` | Free-swing fit good (held-out RMSE 0.23–0.31 rad) |
| 11-06 21:29 → 11-07 03:00 | LQR on hardware: ~14 presets (`lqr_hw_01..18`, `lqr_calibrated`, `lqr_swing`, `lqr_coupling_delay`) | First pass 11-07 02:57. Reference run `030202` |
| 11-07 14:00 → 16:20 | Shared hand-off layer; transfer to MPC, iLQR, PID | MPC pass 15:56, iLQR 16:10, PID 16:16. Tags `*-hw-pass-20261107` |
| 11-07 16:30 → 21:00 | Standalone RL on rig. SAC sim-to-real fight: variants A–N, R25/R50, V1/V2, W1/W2, dither measurements | 0 passes. Many simulator theories rejected |
| 11-07 21:00 → 22:00 | On-rig SAC learner; **P1** (V1 actor pretrained to imitate LQR) | P1 passed (`215833`), later **excluded** as not reward-learned |
| 11-07 22:30 → 11-08 04:30 | Reward-only SAC curriculum (Sol's track). Sim pilots, LQR-placed starts, versions v1→v6b. Encoder fault night (23:30 → ~03:00) | v3b best hold; v6b first reward-only full-task episode in a curriculum batch. 0 official passes |
| 11-08 16:00 → 11-09 00:00 | "Dead cycle": v7–v9, online SAC session s01, smooth SAC (sm*), replay-fit of the simulator | No progress. Filmed failures of PPO, SAC v6b and DQN for the report (21:18–21:20) |
| 11-09 ~00:00 → 00:42 | Everything fails, LQR included (~20 runs). Diagnosis | **Pendulum loose on its shaft.** Re-secured at 00:42 |
| 11-09 00:46 → 01:35 | Rig-noise training (`rn*`), PPO spin-exploit fix (`ec*`) | **SAC rnA** pass 00:46/00:47, **DQN rn05** 01:02/01:03, **PPO** ppo_long 01:10, ppo_ecB 01:23, **ppo_ecG** 01:35 |
| 11-09 morning | Report: held-out stress / near-upright / learning-curve evals, claim-by-claim audit (9 fixes) | Report finalized (`0fdec78`, `0e65b4f`) |
| 11-09 14:10 → 16:01 | Demo prep / demo-format runs: 15 s re-checks, then **30 s runs** of rnA, LQR and ppo_rn05 | rnA re-checks 0/3 at 15 s (best 9.14 s); at 30 s rnA held up to **28.14 s** (`160034`, passes the final-10 s check). Whether any of these were the graded demo: *(unclear)* |

---

## 3. What happened, phase by phase

### 3.1 October: simulator work before the rig existed
Sean implemented the seven controllers on the starter model and trained PPO/SAC/DQN on GPU (`gpu_rl_20261016`). uuuu added robustness sweeps over perturbed rigs and an LQR input-delay model (state augmented with past actions, `ed81574`). SAC got a continuation and robustness run (`sac_robustness_20261016`), then randomized-physics and failure-focused variants (10-22). The focused SAC became `docs/results/sac_focused_policy.zip`, **the swing-up network every classical controller later used on hardware**. Details: [`logging.md`](../../logging.md), [`reproducing-experiments.md`](../reproducing-experiments.md).

### 3.2 Late October: getting the hardware to move
The first power-on knocked the Nano off USB and made the brick beep: the 12 V polarity was reversed (re-soldered 10-23). The AS5600 encoder held I2C SDA low whenever the Nano reset mid-read, for example after a firmware upload. The standing fix: **unplug USB about 5 s before preflight**. The motor only shook in place because its coils were split across the driver's A/B pins. Claude first blamed a burned DRV8825 and was wrong. On 10-29 there was barrel-jack pin confusion and many +/− swaps, which Claude told us to stop. How the motor and power were finally fixed before 11-06 is not recorded *(unclear)*.

### 3.3 11-06: system identification
Tare, three 10 s free swings from about 107° for fitting (`sysid_train_01`), three held-out swings (`sysid_holdout_01`, `sysid_validation_01`), then driven tests: acceleration step and chirp (`accel_*`), motor envelope (`envelope_*`, `sysid_motor_envelope`) and a coupling fit. Results: ωₙ 12.27 rad/s, ζ 0.0101, J 4.65e-5 kg·m² (+4.6% vs CAD), Coulomb plus viscous friction. Held-out RMSE 0.23–0.31 rad, mostly a 3% period drift. **Free swings do not identify delay or motor behaviour.** Those came later from balance logs: 15 ms arm, ~23–25 ms pendulum. An earlier session the same evening is in `before_changes_20261106_211040`. Details: [Report §3](../../Report.md), [system-identification.md](../system-identification.md).

### 3.4 11-06 night: the LQR saga
About 14 presets in one night ([configs/experiments/README.md](../../configs/experiments/README.md) has the full table):
energy swing-up (no hold) → tighter capture (2.24 s) → arm-guarded capture (0.16 s) → robust/adaptive/filtered (≤ 1.18 s) → **SAC swing-up** (crossed upright too fast) → **approach braking** at 4.5 rad/s (1.18 s) → arm recentering → velocity observer (2.36 s) → calibrated 27 ms delay, coupling ×1.1 (6.26 s) → **observed + 25 ms pendulum delay = `lqr_my_rig_coupling_delay.json`: 3 of 16 passed**. First pass 02:57 on 11-07. Codex did this tuning with crasni on the rig. Claude then verified on 11-07 that the current code replays all three passes bit-exactly, and added `tests/test_hardware_replay.py` and the `lqr-hw-pass-20261107` tag.

*Run dirs:* `lqr_hw_*`, `lqr_sim_*`, `lqr_calibrated_*`, `lqr_swing_*`, `lqr_coupling_delay_*` (50 runs from 11-07 to 11-09; the later ones are reference and health checks), `lqr_hardware_pass_20261107_025738.tar.gz`.

### 3.5 11-07 afternoon: one hand-off, four controllers
Claude refactored the swing-up and hand-off (SAC swing-up, approach braking, capture gate, recentering, observer, release at 0.55 rad, arm limiter) into an **identical copy inside each of `lqr.py`/`mpc.py`/`ilqr.py`/`pid.py`**, since the rules want each algorithm in its own file. `tests/test_handoff_sync.py` keeps the copies equal.
- **MPC:** low-`r` (r = 3) and strict-catch presets looked much better in a disturbance simulator (18/20 vs 4/20) but **did worse on the rig** (≤ 1.44 s, 0/10 strict catch). We kept r = 30, which passed (`155613`). Lesson: the simulator could design but not *rank*.
- **iLQR and PID:** "`lqr_match`" presets make them send nearly LQR's command near upright. iLQR passed at 16:10, PID at 16:16.
- Grey areas flagged after reading all 27 slides: PID gains derived from LQR, and the SAC network as a classical swing-up. Both were judged allowed and disclosed in the report.

*Run dirs:* `mpc_*`, `ilqr_*`, `pid_*`, `transfer_check_*`, `custom_robustness_*`, `report_sim_eval`, `seed0_local`.

### 3.6 11-07 evening: standalone RL meets the rig
The merged rig-tuned PPO/SAC/DQN (`*_long`, trained with a 9 rad/s² random-disturbance surrogate) all failed: SAC best 1.50 s, PPO 1.04 s, DQN never got near upright. Then came a sequence of simulator fixes, each built to explain past failures, **each failing on the rig**:

| Variant (dir) | Theory | Rig result |
|---|---|---|
| `sac_real_starts/A..I`, `sac_A_*` | Train from real capture states | fail |
| `sac_D_*` | 30 ms delay | disproved on rig |
| `sac_F_*` | Command-smoothness penalty | fail |
| `sac_residual/J..N`, `sac_J_*` | Learned residual-dynamics simulator | J never swung above 41° ("COM_PLETE FAILURE") |
| `sac_finetune/R25, R50`, `sac_R50_*` | RLPD mixed replay, 25/50% real transitions | over-swung |
| `sac_finetune/V1, V2`, `sac_V1_*` | Velocity-reading noise 0.4–1.2 rad/s | 0/4 (but **V1 became the parent of every later SAC**) |
| `sac_finetune/W1, W2`, `sac_W*` | Arm lag 10–24 ms plus stepper kick | 0/6 |
| `custom_v1raw/v1matched/sacraw_*` | V1/SAC as the *balance* law inside LQR's hand-off (diagnostic, via `custom`) | fail (≤ 1.1 s) |
| `custom_lqrdither_*` | LQR plus known dither, to measure lag | measured ~1 extra tick of pendulum lag (measurement only) |

Measured gaps ([sac-hardware-pass.md](../sac-hardware-pass.md)): from the same 96 real capture states the simulator holds 95–98%, the rig 43% for LQR and **5% for SAC**. The stepper keeps 0.2–0.9 rad/s of residual speed after reversals. Encoder jitter is 0.6° RMS vs 0.045° in sim. The real gap was **catching**: LQR caught 10/21 slow arrivals, SAC 0/24.

**P1.** An on-rig learner (`scripts/rig_sac.py`, `rig_sac_pilot_*`, 2.2 GB of checkpoints) degraded the policy. Pretraining V1's actor to imitate LQR's real balancing commands (`sac_finetune/P1`, `scripts/pretrain_actor.py`) **passed the official runner** (`20261107_215833`, 11.7 s, about 1-in-3 rate). We later **chose not to count it**: its balancing came from demonstrations, not reward.

### 3.7 11-07 → 11-08: reward-only SAC curriculum (Sol's track)
Rules: start from V1; no behaviour cloning; no LQR commands in training; immutable versioned checkpoints; deterministic rig evals decide. A custom "lift" pump for near-upright starts stalled on the rig, so on crasni's suggestion **LQR was used only to place the pendulum** (`lqr_balance` stage). Its ticks never enter training.

Structure in `runs/curriculum/`: **`b*` = data-collection batches** (stochastic policy), **`e*` = deterministic evaluation batches**, **`s01_*` = online session**, `versions/` = immutable policies. Sim pilots are in `curriculum_sim/pilot_a, pilot_b`.

**SAC version lineage** (all from `sac_finetune/V1`; parent → child):
```
V1 ─┬─ v1, v2, v2_u250..u5000   offline SAC on rig ticks only → too little data, rejected on rig
    └─ v2m1, v2m2                50/50 sim + rig-tick updates → v2m2 held 6 s cap 4/4 on lqr_balance, 0/10 official
         ├─ v3                   all rig data → broke swing-up on rig
         ├─ v3c
         └─ v3b                  rig ticks within 45° → best reward-only catcher (frozen); 0/7 official
              ├─ v4, v4b         overspeed penalty → v4 stopped swinging up; rejected offline
              ├─ v5              catch oversampling, no anchor → broke swing-up
              ├─ v6, v6b         + SWING-UP ANCHOR (actions on rig swing-up states pulled to parent)
              │    └─ v6b_runner → first reward-only full-task episode (e10 ep0, 12 s); 0/3 official
              │         ├─ s01e15, s01e17, s01e49   online session s01 (rig_sac_online.py), 60 eps, 1 lucky pass, no trend
              │         ├─ sm2, sm5, sm15           smooth-command penalty → smooth in sim, jumpy on rig, fail
              │         └─ rnA, rnB, rnC            RIG-LIKE HEAVY-TAILED READING NOISE → rnA PASSED 2× (11-09)
              ├─ v7, v7b         → balanced into the arm guard
              │    └─ v9         + arm-guard reward term (crasni's idea) → no change
              └─ v8              mirrored rig data (catches were one-sided) → no change
```
`*_runner` copies exist because checkpoints pickled classes from `scripts/` and `pendulum-lab hardware` could not load them.

Key analysis: after a Codex critique, a catch-ending study (`scripts/catch_endings.py`) showed **arrival speed is not what separates failed catches from good ones**. 14/18 lost catches failed with the arm free and the limiter idle, within about 0.6 s. The weak spot is the ~1 s catch transient. A replay fit ([sim-fit-upright.md](../sim-fit-upright.md)) showed the simulator's average physics is right near upright, but its errors concentrate on **abrupt command reversals**, which SAC makes twice as often as LQR. Full log: [sac-reward-curriculum.md](../sac-reward-curriculum.md), [sac-improvement-proposal.md](../sac-improvement-proposal.md).

### 3.8 11-09 00:00: the loose pendulum
From about midnight every controller failed, LQR included (about 20 runs, best 4.5 s), and the balance tilt jumped from run to run. The tare read 0.00°, so it was not a zero shift. Replay mismatch rose from 5.8% to 27.3% of segments, and swing-up peaks had been drifting up for three days. crasni asked "could the pendulum have come loose?". It had. Re-secured at 00:42, LQR was healthy again (7.7 s hold). **Runs from the degraded period (e15–e17, sm*, early rnA) are suspect, and so may be many earlier "sim-to-real" failures.**

### 3.9 11-09 00:46 → 01:35: all three learned methods pass
- **SAC rnA** (v6b fine-tuned in sim with Student-t reading noise, 0.6°/0.6 rad/s RMS as measured on the rig): passes `004642` (13.18 s) and `004713` (13.12 s, filmed). It also re-caught after pokes.
- **DQN dqn_rn05** (from `dqn_long`, half-strength noise plus 6–20 ms delay): 2/2, `010258`, `010341` (13.56 s, longest of all). Fragile in held-out sim stress (22/32).
- **PPO.** The unchanged `ppo_long` passed once after the repair (13.50 s), which suggests the loose pendulum explains part of the earlier failures. `ppo_rn05` exposed a **spin exploit**: under the lab reward spinning earns about 11/tick vs 5 hanging, and it spun at 37 rad/s on the rig. An **energy-cap** shaping term (`ppo_ec*`) fixed it. Lineage: `ppo_long → ppo_ecB → ppo_ecE → ppo_ecG`; ecG adds random delay, arm lag, coupling and real-state starts. Passes: ecB `012337` (13.52 s), **ecG `013508`** (report run).

Lineage files: `docs/results/rig_rl_disturbance/{sac_rnA_reward_only,dqn_rn05,ppo_ecG}/LINEAGE.md`. Hardware runs: [rl-hardware-passes.md](../rl-hardware-passes.md).

### 3.10 11-09 afternoon: demo preparation runs
After the report audit, the rig was used again from 14:10 (all reloaded from the promoted `docs/results/` checkpoints):
- 14:10–14:12, 15 s re-checks: rnA held 5.88 s and 9.14 s, dqn_rn05 5.18 s. **None passed.** Both had passed at 00:46–01:03 the same night, which again shows how variable the rig is.
- 15:22–16:01, **30 s runs** (the demo format is two 30 s runs): rnA ×7 (holds 2.0–22.1 s, then **28.14 s** in `160034`), LQR `152405` (11.26 s), and ppo_rn05 ×4 ("the PPO that went crazy", 2.1–8.2 s; spin demonstration).
- These runs are in `runs/` and in `runs_index.md`. They are **not** in the report's attempt counts. Which runs, if any, were the live demo is not recorded *(unclear)*.

### 3.11 11-09 morning: report audit
New results: held-out stress batch (seeds 92000+), near-upright starts (93000+; the classical controllers fail 5/20, all on one side, revealing a one-sided capture region), learning curves and checkpoint stress evals (DQN's deployed final checkpoint scores 0/10 in the combined case while neighbours score 10/10). A "held-out hanging seeds" idea was dropped as meaningless, since the hanging start is exact. A claim-by-claim audit found and fixed 9 errors (wrong `environment.txt`, a rig-file hash broken by the Group08 rename, stale hold claims and others).

---

## 4. Key decisions and why

| When | Decision | Reason | Driven by |
|---|---|---|---|
| 10-23 | Unplug USB 5 s before every preflight | AS5600 I2C lock-up after a mid-read reset | Claude (diagnosis) |
| 11-07 | Freeze the LQR pass; refactor only behind a bit-exact replay test | Evidence must stay reproducible for the report | crasni / Claude |
| 11-07 | One shared hand-off, copied into each controller file | Transfer the LQR win without breaking the "own file" rule | Claude |
| 11-07 | Keep MPC r = 30 despite simulator preferring r = 3 | Hardware said r = 3 was worse | Hardware evidence |
| 11-07 | No residual RL or hybrid for graded RL | Learned methods must swing up and balance alone | Rules (Claude read them) |
| 11-07 | Do not count P1 | Balancing learned by imitation, not reward | Team; Sol's reward-only track |
| 11-07 | LQR only *places* the pendulum for curriculum starts | Proven on the rig; its commands never enter training | crasni |
| 11-08 | Swing-up anchor in updates | Updates kept breaking the rig swing-up invisibly to the sim | Claude |
| 11-08 | Stay strictly within SAC for the SAC result | Purity of the learning-based claim | crasni |
| 11-09 | Same rig-noise recipe, trained natively per algorithm | rnA worked; each algorithm keeps its own learner | crasni asked; Claude checked lineage |
| 11-09 | Count every attempt; exclude the ~20 loose-pendulum LQR runs from pass counts | Honesty; those reflect a mechanical fault | Claude / crasni |
| 11-09 | Demo candidates: rnA (pure SAC, re-catches after pushes), rn05, ppo_ecG, LQR | Best evidence per method | crasni |


---

## 5. Dead ends and lessons

**Dead ends** (each "matched past failures" yet failed to predict the next policy):
energy-shaping swing-up on hardware (≤ 2.24 s); MPC low-r and strict-catch; the 30 ms-delay theory; residual-dynamics simulator; velocity noise alone; arm-lag plus stepper-kick model; on-rig online SAC with imitation (degraded P1); the custom lift pump; offline-only updates (v1/v2); unfiltered rig data (v3); overspeed penalty (v4); catch oversampling without an anchor (v5); mirrored data (v8); arm-guard reward (v9); online session s01; smooth SAC (sm*); the "arriving too fast" hypothesis; PPO spin fixes C/D/E; the "zero offset shifted" diagnosis.

**Lessons**
1. Run a quick LQR reference run at the start of every rig session and watch the balance tilt. Mechanical drift looks exactly like a sim-to-real gap.
2. Measure sensor noise, actuator stop behaviour and delay **before** training, and put them in the training environment from day one.
3. Validate the simulator on what matters (replay real near-upright states), not just free swings.
4. A simulator that fits past failures is not validated until it predicts a *new* policy's rig result.
5. Check rewards for exploits (PPO spinning) before long training runs.
6. Select value-based checkpoints by stress test, not "take the last one" (DQN).
7. Small eval batches (4 episodes) cannot tell versions apart at a ~1-in-10 pass rate; budget for that.
8. Keep every attempt, including faults. It made the analysis possible and the report honest.

---

## 6. Working with the AI

- **Division of labour:** crasni runs the rig; the AI writes code, gives exact commands, and reads `runs/` afterwards. **The AI must never start rig runs itself** ("do NOT RUN ON YOUR OWN BUDDY, JUST READ THE RESULT", 11-08).
- **No `Co-Authored-By` or AI attribution in commits, ever** (11-09). Four pushed commits still carry it (`051bc5e`, `3d07a5c`, `0fdec78`, `0e65b4f`); history was deliberately not rewritten.
- crasni switches between Traditional Chinese and casual English and wants simple step-by-step explanations when overloaded. Wants sources ("你怎麼知道的 來源是什麼", "how do you know, what's your source") and no filler ("do not find for the sake of generating reply to me").
- crasni's domain sense often corrected the AI: the DRV8825 was not broken, use LQR for placement, the pendulum is loose, v7 is hugging the guard.
- Several AIs cross-checked each other: Codex's critique overturned Claude's "arrival speed" theory; Sol shaped the reward-only rules.
- A teammate edits `Report.md` on `main`: fetch first, and put findings in separate docs.


---

## 7. Hardware notes for next time

| Symptom | Cause found | Fix |
|---|---|---|
| Nano drops off USB at 12 V on, brick beeps | Reversed 12 V polarity | Re-solder; the DRV8825 has no reverse protection |
| TestEncoder hangs, frozen pendulum reading, fault bits=2 at start | AS5600 holds SDA low after a mid-read reset | Unplug USB about 5 s before preflight |
| Recurring `NonFiniteFeedback` bits=2 mid-run (11-07 23:30 → 11-08 03:00) | I2C data wiring suspected; magnet strength fine | Fixed by crasni *(how: unclear)* |
| Motor shakes in place | Coils split across A/B | Re-pair the coils |
| Holds shorten, tilt jumps run to run, swing-up peaks drift | **Pendulum loose on shaft/magnet** | Re-secure; check first |

---

## 8. Glossary of run names

| Name | Meaning |
|---|---|
| `*_YYYYMMDD_HHMMSS` | Official `pendulum-lab hardware` run (start time, local) |
| `lqr_hw_NN`, `lqr_sim_*` | Early LQR hardware and sim runs, one per preset ([table](../../configs/experiments/README.md)) |
| `coupling_delay` | The passing LQR preset: observer plus 15 ms arm / 25 ms pendulum delay model |
| `transfer`, `lqr_match` | MPC/iLQR/PID presets copying LQR's hand-off / LQR's commands |
| `strict_catch`, `low_r` | Rejected MPC variants |
| `*_long` | Rig-tuned RL policies from `rl/rig-training` (disturbance continuation) |
| `sac_real_starts/A–I`, `sac_residual/J–N` | 11-07 SAC sim-to-real variants (real starts; residual-dynamics simulator) |
| `R25`, `R50` | RLPD mixed replay with 25 / 50% real transitions |
| `V1`, `V2` | Velocity-reading noise 0.4–1.2 rad/s (V2 also 25% real). **V1 = root of reward-only SAC** |
| `W1`, `W2` | Arm lag plus stepper-kick simulator |
| `P1`, `P1alt` | V1 actor pretrained to imitate LQR (passed; not counted) |
| `custom_v1raw/v1matched/sacraw` | V1 or SAC as balance law inside LQR's hand-off (diagnostic) |
| `custom_lqrdither` | LQR plus known dither for lag identification |
| `curriculum/bNN_*` / `eNN_*` / `s01_*` | Data-collection batch / deterministic eval batch / online session; suffix = version and stage (`lqrbalance`, `hanging`), `det` = deterministic, `_rN` = repeat |
| `v1…v9`, `v2m2`, `v3b`… | Reward-only SAC versions (lineage in §3.7); `m` = mixed sim/rig updates |
| `*_runner` | Same weights re-saved so the official runner can load them |
| `s01eNN` | Online session s01, checkpoint after episode NN |
| `sm2/sm5/sm15` | Smooth-command penalty weights |
| `rn*` | **R**ig-like reading **n**oise (Student-t); `rn05` = half strength; `rnA/B/C` = SAC variants |
| `ec*` | PPO with **e**nergy **c**ap (anti-spin); `ecG` = report PPO |
| `archive/` | Runs with no usable result (faults, cancels); see `metadata/archive/README.md` |

---

## 9. What was deleted from `runs/` and what remains

`runs/` was never in git. On 2026-11-10 the `checkpoints/` directories and `*.pkl` replay buffers (8.45 GB of 9.03 GB) were deleted. Before deletion each one's path, size and sha256 were recorded in [`file_manifest.csv`](file_manifest.csv) (`prunable=1`). **Kept:** every final `policy.zip` / `actor_weights.pt`, all episode and transition `.npz` data, all metadata and logs, and the checkpoints listed in [`kept_referenced.txt`](kept_referenced.txt) (used by `tests/test_checkpoint_portability.py` and `logging.md`; `sac_residual/J2` has no final policy). Every policy that any run loaded is a kept final `policy.zip`; no hardware run deployed a deleted intermediate checkpoint.

**Already missing before this cleanup:** some paths cited in older docs and JSONs, e.g. `runs/disturbance_rl_v2/*`, `runs/gpu_rl_20261016/*/checkpoints/*`, `runs/sac_randomized_20261022`, `runs/sac_focused_20261022` checkpoints, `runs/dqn_gpu_new`. They were removed in earlier cleanups. The policies that mattered were promoted to `docs/results/` (e.g. `docs/results/rig_rl_disturbance/*`). The committed evidence for the report lives in `docs/results/hardware/` regardless.

## 10. Open questions
- Live demo outcome and the final grade are not recorded here. The 11-09 afternoon 30 s runs (§3.10) may be the demo or rehearsals.
- How much of the pre-repair failures were the loose pendulum vs real sim-to-real gaps. Old SAC versions (v3b, v6b, s01e49) were never retested on the repaired rig.
- Why rig-noise training works (robustness to jitter vs something else) is not proven.
- PPO's spin exploit is mitigated, not eliminated, under the lab reward.
- The classical hand-off's one-sided capture region (near-upright starts) was found but not investigated.
- Who "Sol" was.
