# How to train reward-only SAC better on our rig (proposal, 2026-11-08)

Context: [sac-reward-curriculum.md](sac-reward-curriculum.md) (v2m2 → v9). Swing-up is
solved (the anchor keeps it in every rig episode since v6b). Balancing is not:
12-episode comparisons give v7 1.80 s, v9 1.82 s, v6b 0/5 official passes. Catches
are lost when the arm runs into the 77° guard while the pendulum leans the other way.

## Requirements this proposal keeps

- **SAC**: SAC's actor, critic and entropy losses; deterministic evaluation from hanging, SAC alone.
- **Reward only**: no LQR commands, no other controllers' data, no P1 weights, no cloning.
  (LQR may *place* the pendulum at the start of a training episode; its ticks are never training data.)
- **Same safeguards**: runner limiter, guards, deadlines unchanged.
- **Resources**: 16-core CPU, no GPU. Measured: 356 SAC gradient steps/s (128×128,
  batch 256, 4 threads), i.e. about 7 updates per rig tick in real time. You run the rig;
  rig time is the scarce resource, so the plan uses one long attended session, not many short rounds.

## What SAC is, and what that means here

SAC learns a critic Q(s, a), "how much reward follows if I do a here", from replayed
transitions, and moves the actor toward actions the critic rates highly, plus an entropy
bonus that keeps it exploring. Four properties matter for us:

1. **It is built to correct itself by trying.** The critic is only right about actions it
   has seen the result of. SAC's actor goes wherever the critic is optimistic, and SAC
   relies on then *trying* those actions to correct the critic. Training on a fixed log
   removes that correction: the actor follows the critic's optimistic guesses into
   untested territory. That is exactly v3 → v9: offline jumps, then surprises on the rig.
2. **Our critic still believes the simulator.** V1's critics were trained for 600k
   simulator steps (UTD 0.25). Our rig updates (40k steps, half of each batch from the simulator)
   barely re-fit them near upright, where the simulator is wrong. Deep RL networks
   overfit early experience and resist later data (*primacy bias*); the standard fix is
   to re-initialise part of the network and keep the data.
3. **Its exploration is white noise, and the rig filters it out.** SAC samples an
   independent Gaussian every 20 ms. The arm is commanded in acceleration, so the
   stepper integrates the noise and successive kicks largely cancel. The move
   SAC never found, a sustained lean back of the pendulum to stop the arm drifting, is a
   slow, 0.5–1 s manoeuvre. White noise almost never produces it; temporally correlated
   noise does.
4. **Sample efficiency is a solved-ish problem, with specific ingredients.** Plain SAC
   learned a real quadruped gait in about 2 hours (Haarnoja 2018). With a normalised
   critic and many updates per real step (DroQ / RLPD), the same robot
   learned in about 20 minutes (Smith 2022), and real manipulation tasks in 25–50 min (SERL 2024).
   We have ~20 min of rig data but used none of those ingredients.

## Proposal: on-rig SAC with the sample-efficiency recipe

One change of method, not another reward tweak. All of it is still SAC.

| Change | Why (property above) | Source | Effort |
|---|---|---|---|
| **New critics with LayerNorm (+ small dropout)**, fitted first on all SAC rig data with the actor frozen | 2: replace the simulator-biased critic; LayerNorm bounds extrapolation to unseen actions (1) | DroQ (Hiraoka 2022), RLPD (Ball 2023), primacy bias (Nikishin 2022) | ~1 h |
| **High update rate, during the episode**: UTD ≈ 5–10 with the existing async learner in `rig_sac.py` | 4: the ingredient behind 20-minute real-robot learning | Smith 2022 (UTD 20), RLPD | already built; tune |
| **Symmetric sampling**: each batch 50% this session / 50% earlier SAC rig data (no LQR logs) | 1: old data for coverage, fresh data so SAC corrects its own mistakes | RLPD | small change to `load_prior` |
| **Pink (colored) exploration noise** in training episodes | 3: produces the slow corrective moves white noise cancels | Eberhard 2023 (ICLR spotlight, tested on SAC); gSDE (Raffin 2021) for real robots | ~1 h |
| **LQR-placed starts** for most training episodes, every 5th a deterministic hanging check | spend rig time where SAC is weak (catch and hold) | curriculum's `lqr_balance` stage | port from `rig_curriculum.py` |
| **Swing-up anchor kept** | keeps the one thing that works | ours (v6b) | port |
| Keep: SAC losses, auto-temperature, γ 0.99, reward incl. arm-guard term, runner limiter | | | |

The 2026-11-07 on-rig pilot that degraded P1 differs on every point that matters:
it used LQR imitation and LQR logs in replay, lr 3e-5, UTD 1, the old critic, white
noise, and 20 episodes.

### How it runs

1. **Offline, no rig (me, ~3 h build + test in the simulator):** implement the five changes in
   `scripts/rig_sac.py`; refit the new critics on all SAC rig data; run the whole
   session against the rig-matched simulator (`--sim --realtime`) to check deadlines
   at the chosen UTD and that nothing collapses.
2. **One attended rig session (you, ~25–30 min, one command):** start from v6b (anchor to
   its swing-up). About 60 training episodes of 8 s from LQR placement, with a
   deterministic 15 s hanging episode every 5th. The learner trains while the episode
   runs; a checkpoint is saved every episode. Stop early if the hanging checks get worse three times in a row.
3. **Selection (me):** pick the best checkpoint by its hanging checks, then **one
   12-episode official-runner evaluation (you, ~5 min)**. Report every attempt.

### What to expect

- Best case: steady improvement within the session, as in the 20-minute papers, ending
  with consistent catches and holds.
- Likely: better than v6b–v9 (which never learned from their own fresh behaviour), but
  not reliably passing; still a reportable result with a learning curve from the rig.
- Main risks: the stepper and the 77° guard make the rig harsher than a quadruped on grass;
  CPU limits UTD to ~7, below the papers' 20; pink noise may need its strength tuned on the rig.

## Not proposed, and why

- **More reward terms or data tricks judged offline** (mirror, guard, overspeed): none of the offline checks predicted the rig.
- **AWAC / IQL**: they replace SAC's actor loss, so the policy would no longer be SAC.
- **Self-imitation / cloning SAC's own best catches**: conflicts with "no cloning".
- **CQL offline pre-training alone**: still offline; known to "unlearn" at the start of online
  fine-tuning (Cal-QL 2023). LayerNorm critics + symmetric sampling get most of the
  benefit without it (RLPD).
- **Another simulator fix**: three validated simulator changes each failed to predict a new policy.

## Sources

- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, 2018 — https://arxiv.org/abs/1812.05905
- Smith, Kostrikov, Levine, *A Walk in the Park: Learning to Walk in 20 Minutes*, 2022 — https://arxiv.org/abs/2208.07860
- Ball, Smith, Kostrikov, Levine, *Efficient Online RL with Offline Data (RLPD)*, ICML 2023 — https://arxiv.org/abs/2302.02948
- Hiraoka et al., *Dropout Q-Functions for Doubly Efficient RL (DroQ)*, ICLR 2022 — https://arxiv.org/abs/2110.02034
- Luo et al., *SERL: Sample-Efficient Robotic RL*, ICRA 2024 — https://arxiv.org/abs/2401.16013
- Nikishin et al., *The Primacy Bias in Deep RL*, ICML 2022 — https://arxiv.org/abs/2205.07802
- Eberhard et al., *Pink Noise Is All You Need*, ICLR 2023 — https://iclr.cc/virtual/2023/poster/11515
- Raffin, Kober, Stulp, *Smooth Exploration for Robotic RL (gSDE)*, CoRL 2021 — https://arxiv.org/abs/2005.05719
- Nakamoto et al., *Cal-QL*, NeurIPS 2023 — https://arxiv.org/abs/2303.05479

## Implementation (2026-11-08): `scripts/rig_sac_online.py`

Built as proposed, as a new script (`rig_sac.py` stays as P1's tool). Defaults:
v6b's actor; new 256×256 critics with LayerNorm and dropout 0.01; 20k critic-only
steps on the 92,785 earlier SAC rig transitions (all curriculum batches); UTD 8 with
8 critic updates per actor update (one actor step per rig tick); 50/50 session/prior
batches; swing-up anchor ≥ 30° ×2; pink noise; arm-guard reward (weight 20 from 45°);
training episodes LQR-placed (`lqr_balance`, 8 s cap, end 0.5 s after a fall); every
5th episode deterministic from hanging, 15 s. The session stops by itself if the swing-up fails in two
evaluations in a row, or the placement fails three times in a row. Checkpoints load in the official runner
(the critics inside the zip are v6b's; the LayerNorm critics sit beside it as `*.critic_ln.pt`).

Two fixes found in real-time simulator rehearsals, before any rig time:

| Rehearsal | Mean Q (−actor loss) | Critic loss | Temperature | Anchor loss |
|---|---|---|---|---|
| 1: actor updated on every gradient step | 245 → 545 | 23 → 63 | 0.17 → 0.33 | 0.10 → 0.25 |
| 2: + 8 critic updates per actor update | 241 → 548 | 22 → 116 | 0.16 → 0.24 | 0.10 → 0.27 |
| 3: + dropout also in the target critics (as in DroQ) | 171 → 261, levelling | 9 → 24 | 0.15 → 0.12 | 0.06 → 0.14 |

The reward's ceiling is about 500 (5 per tick ÷ (1 − 0.99)), so rehearsals 1 and 2 were
overestimating. Offline on the same data, mean Q on earlier rig states went
111 → 477 still rising without target dropout, 91 → 233 levelling with it, 79 → 168 with 4 critics
(2× slower; not used). Ticks stayed on schedule with the learner at full load (worst 24.7 ms;
the limit is 40 ms over). The simulator's from-hanging scores are not used to judge learning: v6b
scores 0.26 s there before any update, while holding up to 12 s on the rig.

## Rig session `s01_v6b_online` (2026-11-08, 60 episodes, completed)

Learner stable throughout: mean Q 174 → 243 (below the ~500 ceiling), critic loss 12–30,
temperature 0.155 → 0.113, anchor loss ≤ 0.15. Swing-up intact in all 12 evaluations
(top at 2.2–4.0 s). 8 LQR placements timed out (no SAC ticks; not policy-related).

| Evaluations (deterministic, hanging, 15 s) | Updates | Longest hold |
|---|---|---|
| ep 4 (v6b, before learning) | 0 | 6.12 s |
| ep 9, 14, 19, 24, 29, 34, 39, 44 | 4k–95k | 1.40, 0.30, 0.10, 1.38, 0.26, 0.56, 0.90, 1.80 s |
| **ep 49** | 105k | **10.90 s, full-task pass** |
| ep 54, 59 | 115k–125k | 0.32, 0.08 s |

Training episodes (LQR-placed, pink noise, 8 s cap): mean longest hold 5.5 s before any
update (episodes 0–6), about 2.8 s afterwards. **No learning trend:** the session did not
improve balancing; one pass in 11 learned evaluations is consistent with v6b's own rate.
The episode-49 actor is kept as `versions/s01e49` (checkpoint at 105,308 updates; the
actor that ran episode 49 was published 100 updates, about 12 actor steps, earlier).
Its zip first carried a learner-only attribute and did not load without `scripts/`;
re-saved with identical weights, and `LayerNormCritics.save` fixed.
