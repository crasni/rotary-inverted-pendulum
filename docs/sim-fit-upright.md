# Fitting the simulator near upright to the rig logs (2026-11-08)

Tool: `scripts/fit_sim_upright.py`. Each test segment starts from a logged rig state
within 20° of upright. The simulator (the rig-matched `SimClient` settings)
replays the exact commands the rig received for 10 ticks (0.2 s), open loop,
and is compared with the logged pendulum angle and arm step count. The upright
instability amplifies measurement noise in the starting state about e^(12t), so each
segment's starting angle (±1.5°) and pendulum speed (±1.5 rad/s) are re-estimated.
The loss is a trimmed mean that drops the worst 5%, so a few touched or stuck episodes
cannot steer the fit. Data: 46 min of SAC rig episodes (curriculum batches and the
on-rig session) and all LQR rig runs.

## Results

1. **The simulator's average near-upright physics is right.** Median error over 0.2 s
   is 0.2–0.4°, below the rig's own angle jitter (about 0.6° rms). No setting of the existing knobs
   (command delay, arm lag, coupling, stepper kick, pendulum friction, gravity torque)
   improves it meaningfully: trimmed error 1.03–1.19° across variants (base 1.18°).
   An earlier result that the 18 ms arm lag was wrong came from a replay seeding bug
   (the lagged arm was started at the step count); fixed, the effect disappears.
2. **Its errors concentrate on abrupt command changes.** Share of segments with
   error > 3°, calm starts (|θ| < 8°, |θ̇| < 1.5 rad/s):

   | Largest tick-to-tick command change in the segment | LQR's commands | SAC's commands |
   |---|---|---|
   | < 0.6 | 0% | 1% |
   | 0.6–1.2 | 1% | 6% |
   | 1.2–2.0 (reversals) | 5% | 18% |

   Over all segments, error > 5° occurs in 3.0% of LQR segments and 16.3% of SAC segments.
3. **SAC commands abrupt changes about twice as often as LQR on the rig.** Near
   upright, a change > 0.6 per tick happens on 24% of ticks for deterministic SAC
   vs 11% for LQR (median change 0.26 vs 0.13).
4. **A softer arm servo is not the mechanism.** Lowering the arm's PD servo bandwidth
   (default ~400 Hz) to 15–60 Hz made every segment class much worse (smooth segments
   3% → 26–52% bad). MuJoCo's servo damps absolute arm speed, which adds a speed-dependent
   lag the rig does not show.

## What this means

The simulator is a valid model of the rig for smooth control (LQR-like command
changes) and an unreliable one for abrupt reversals. SAC, trained in simulation,
learned catches that use those reversals. That is consistent with every earlier observation:
LQR transfers, SAC catches do not, and simulator holds are optimistic for SAC.

## Smooth SAC (`scripts/train_smooth_sac.py`)

Fine-tuned from v6b in V1's training simulator (reward only, no rig data) with an
extra penalty `k · exp(−θ²/(30°)²) · (a_t − a_{t−1})²` near upright; swing-up anchored
to v6b (rig states ≥ 45°, ×2); 150k simulator steps, lr 1e-4, about 4 min each.

| Version | k | Near-upright ticks with change > 0.6 (rig-matched sim) | Rig-matched sim, hanging 15 s | Standard sim | Swing-up corr. with v6b |
|---|---|---|---|---|---|
| v6b | – | 8.7% | 1/6 | pass | 1.00 |
| s01e49 | – | 10.6% | 0/6 | – | 0.93 |
| **sm2** | 2 | 0.1% | 5/6 | pass (17.3 s) | 0.99 |
| **sm5** | 5 | 0.0% | 6/6 | pass (17.8 s) | 0.99 |
| sm15 | 15 | 4.1% | 0/6 | – | 0.98 |

The same metric on the rig: deterministic SAC 24%, LQR 11%. Whether sm2/sm5 stay
smooth on the rig (where readings are noisier) is the first thing to check.

### Rig result (`e16`, 2026-11-08): smooth in simulation, not on the rig

| | Large changes near upright, sim | Same, rig | Lag-1 autocorr. of command change (rig) | Rig, hanging, deterministic |
|---|---|---|---|---|
| sm5 | 0.0% | 37.6% | −0.16 | 12 episodes, 0 passes, mean 0.70 s |
| sm2 | 0.1% | 24.3% | −0.24 | 8 episodes (one batch cancelled), 0 passes, mean 2.1 s |
| v6b / v7 | 8.7% | 25.5% | **−0.48** (tick-to-tick chatter) | earlier: 0–1 passes per 12–20 |
| LQR | – | 11.4% | −0.23 | passes |

Swing-up intact in every episode (top at 2.2–3.2 s). The penalty removed v6b's
tick-to-tick chatter (autocorrelation −0.48 → −0.16/−0.24, LQR-like), but the
catches did not improve: sm5 is worse than v6b. Its large changes on the rig come
from failing catches (the pendulum falls through the 20° band), not from chatter.
The command-reversal error explains part of the simulator's optimism, not SAC's
failure to catch. Not pursued further.

## Rig-noise SAC and the loose pendulum (2026-11-09)

Perturbed-simulator gate (`scripts/robust_gate.py`): the "fitted" family ranks LQR
first (6.0 s mean hold) but does not rank SAC versions as the rig does
(Spearman 0.02 over 8 versions). It did show SAC is far more sensitive than LQR to
reading noise, so three candidates were trained from v6b (`train_smooth_sac.py`):
**rnA** with rig-like heavy-tailed reading noise (Student-t, 0.6° / 0.6 rad/s rms),
rnB adding an observation-smoothness loss, rnC adding gamma 0.995. rnB and rnC
lost the catch in simulation; rnA passed the standard simulator.

From about 00:00 on 2026-11-09 every controller failed, LQR included (about 20 runs,
best 4.5 s), with the balance tilt changing from run to run (−0.7° to +2.9°). The
pendulum had come loose relative to its shaft and magnet, confirmed by the user. The
deterministic swing-up's second peak had crept up from about 59° (Nov 7) to about 68°,
so the loosening may have been developing over the earlier sessions. After re-securing
it, LQR held 7.7 s with −0.4° tilt.

Official runner, rnA, after the fix: 20261109_004522 encoder fault (no ticks),
004614 held 12.78 s, **004642 pass (13.18 s)**, **004713 pass (13.12 s, filmed)**.
Promoted to `docs/results/hardware/sac/`. Checkpoint committed unchanged as
`docs/results/rig_rl_disturbance/sac_rnA_reward_only/policy.zip` (lineage:
rnA → v6b → … → V1, all reward-only SAC). The earlier rnA attempt 20261109_000758
(1.2 s) ran with the loose pendulum and is not promoted. SAC results from 2026-11-08
evening and 2026-11-09 before 00:42 (e15–e17, sm5/sm2, rnB/rnC) are suspect for
the same reason.

## Same training environment for PPO and DQN (2026-11-09)

`train_rig_rl.py --rig-noise X` adds the same heavy-tailed reading noise to the
PPO/DQN fine-tuning environment (each still trained by its own SB3 learner, from
its own `*_long` checkpoint, `--delay-range 0.006 0.02 --disturbance-rms 6`).

| Version | Rig noise | Standard sim | Official runner |
|---|---|---|---|
| ppo_rn | 1.0x, 600k steps | pass (18.2 s) | 20261109_010227 fail (0.18 s; swing-up reached the top, no catch) |
| dqn_rn | 1.0x, 600k steps | **fail** | not run |
| **dqn_rn05** | 0.5x, 400k steps | pass (17.7 s) | **20261109_010258 pass (13.06 s), 20261109_010341 pass (13.56 s, filmed)** |

DQN's earlier rig attempts (dqn_long) held 0.06 s and 0 s. Promoted, with
checkpoint and lineage in `docs/results/rig_rl_disturbance/dqn_rn05/`.

**Disturbance test, not a graded attempt:** `runs/sac_rnA_20261109_011649` (not promoted).
The user poked the balancing pendulum 4–5 times with a light object (2 pokes too hard). rnA held 7.36 s
from its first catch (1.9 s), and after each knock-down swung back up and caught again
(9.7 s, held 2.74 s; 12.8 s; 14.1 s) instead of spinning.
