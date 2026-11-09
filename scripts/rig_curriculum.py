"""On-rig SAC curriculum with assisted near-upright starts, learned from reward only.

Goal: a SAC policy whose balancing is learned from SAC's own reward on the rig,
starting from V1 (SAC-only lineage, matching actor and critics). No behaviour
cloning, no LQR commands, no earlier rig logs of other controllers.

Three separate steps, so every collection batch uses one fixed checkpoint and
every update writes a new, immutable version (rollback = use an older one):

  collect   one attended rig session (or --sim) with a fixed checkpoint. Each
            episode starts from the runner's hanging start check; in assisted
            stages a scripted lift (pendulum_lab.rig_reset.LiftController)
            then swings the pendulum into a near-upright window and SAC takes
            over on that tick. The lift is a reset tool: its ticks are saved
            for inspection but never become training data. Stochastic actions
            for training batches, --deterministic for evaluation batches.
  update    offline SAC updates (critic, actor, entropy: the ordinary SAC
            losses) on the policy ticks of the given batches; writes
            <output>/policy.zip and version.json with the full provenance.
  report    balance duration, arm drift, limiter intervention and full-task
            success per batch, the measures that decide whether to continue.

Stages are in configs/rig_curriculum.json: balance -> capture -> recover ->
mixed (adds hanging starts to keep swing-up). The graded evaluation is always
`collect --stage hanging --deterministic`: SAC alone from hanging.
"""
import argparse
import glob
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from pendulum_lab.rig_session import AssistedStart, RigSession, SessionStop, SimClient, save_episode
from rig_sac import PARAMS, SAC_CONFIG, Actor, episode_transitions, reward_weights

STAGES = Path("configs/rig_curriculum.json")
MAX_LIFT_FAILURES = 3
# Sign of each observation-frame entry under the left-right mirror
# (arm, sin theta, cos theta, arm_vel, pen_vel, last_action).
MIRROR_FRAME = (-1.0, -1.0, 1.0, -1.0, -1.0, -1.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_limits():
    config = json.loads(SAC_CONFIG.read_text())["arm_command_limits"]
    return config["delay_s"], config["speed_rad_s"], config["travel_rad"]


def check_lineage(checkpoint):
    """Refuse checkpoints made by imitation (pretrain_actor.py writes pretraining.json)."""
    directory = Path(checkpoint).parent
    if (directory / "pretraining.json").exists():
        raise SystemExit(f"{checkpoint} was trained by imitation; this curriculum must start from SAC-only weights")
    version = directory / "version.json"
    if version.exists():
        parent = json.loads(version.read_text())["parent"]
        check_lineage(parent)


LQR_CONFIG = Path("configs/lqr_my_rig_coupling_delay.json")
LQR_SWING_UP = Path("docs/results/sac_focused_policy.zip")  # the checkpoint the LQR rig runs used


def lqr_lifter():
    """The rig-proven LQR preset (its own swing-up, catch and hold), used only to
    place the pendulum. Its ticks are never training data."""
    from pendulum_lab.controllers import load_controller
    return load_controller("lqr", PARAMS, LQR_CONFIG, LQR_SWING_UP)


def start_for(stage, rng, lifter=None):
    if rng.random() < stage.get("hanging_fraction", 0.0):
        return None
    if stage.get("lift") == "lqr":
        return AssistedStart(target_energy=math.nan, theta_rad=math.radians(stage["theta_deg"]),
                             pen_speed=stage["pen_speed"], arm_rad=math.radians(stage["arm_deg"]),
                             arm_speed=stage["arm_speed"], lifter=lifter,
                             hold_s=float(rng.uniform(*stage["hold_s"])), timeout_s=stage.get("timeout_s", 40.0),
                             end_after_fall_s=stage.get("end_after_fall_s"))
    low, high = stage["target_energy"]
    return AssistedStart(target_energy=float(rng.uniform(low, high)),
                         theta_rad=math.radians(stage["theta_deg"]), pen_speed=stage["pen_speed"],
                         arm_rad=math.radians(stage["arm_deg"]), arm_speed=stage["arm_speed"],
                         tolerance=stage["tolerance"])


def open_session(args, limits):
    if args.sim:
        knobs = {}
        if getattr(args, "sim_knobs", None):
            fitted = json.loads(Path(args.sim_knobs).read_text())["fitted"]
            knobs = {"arm_lag_s": fitted["arm_lag_s"], "kick": fitted["kick"], "delay_s": fitted["delay_s"],
                     "coupling": fitted["coupling"], "friction": fitted["friction"], "gravity": fitted["gravity"]}
        client = SimClient(PARAMS, seed=args.seed, **knobs)
        clock = [0.0]
        session = RigSession(client, arm_limits=limits, episode_seconds=args.episode_seconds,
                             clock=lambda: clock[0], sleep=lambda d: clock.__setitem__(0, clock[0] + max(d, 0)),
                             log=lambda *_: None)
        return client, session, (lambda _: "RUN")
    from pendulum_lab.serial_client import LowLevelClient
    client = LowLevelClient(args.port, timeout=0.1).__enter__()
    return client, RigSession(client, arm_limits=limits, episode_seconds=args.episode_seconds), input


def collect(args):
    check_lineage(args.checkpoint)
    stages = json.loads(STAGES.read_text())["stages"]
    stage = stages[args.stage]
    limits = arm_limits()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
            "stage": args.stage, "stage_parameters": stage, "deterministic": args.deterministic,
            "episodes_requested": args.episodes, "episode_seconds": args.episode_seconds,
            "seed": args.seed, "target": "sim" if args.sim else args.port, "arm_limits": limits,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (args.output / "batch.json").write_text(json.dumps(meta, indent=2) + "\n")
    actor = Actor(args.checkpoint, limits, deterministic=args.deterministic)  # fixed for the whole batch
    weights = reward_weights()
    rng = np.random.default_rng(args.seed)
    lifter = lqr_lifter() if stage.get("lift") == "lqr" else None
    client, session, confirm = open_session(args, limits)
    reports, stop_reason, failures = [], "completed", 0
    try:
        session.start(confirm=confirm)
        for index in range(args.episodes):
            start = start_for(stage, rng, lifter)
            episode = session.run_episode(actor, start=start)
            report = save_episode(args.output, index, episode, {
                "episode": index, "deterministic": args.deterministic, "assisted": start is not None,
                "lift": stage.get("lift", "energy") if start else None,
                "hold_s": start.hold_s if start and start.lifter else None,
                "target_energy": start.target_energy if start and not start.lifter else None})
            if episode.rows:
                np.savez_compressed(args.output / f"transitions_{index:03d}.npz",
                                    **episode_transitions(episode, weights))
            reports.append(report)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            if episode.stop_reason == "lift_timeout":
                failures += 1
                print(f"episode {index}: lift did not reach the window in time", flush=True)
            else:
                failures = 0
                h = episode.handover
                start_text = (f"from {math.degrees(h[1]):+.0f} deg, {h[3]:+.1f} rad/s, arm {math.degrees(h[0]):+.0f} deg"
                              if h else "from hanging")
                print(f"episode {index} {start_text}: longest balance {report['longest_balance_s']:.2f}s, "
                      f"holds>1s {report.get('holds_over_1s')}, success {report['success']}, "
                      f"limiter {report.get('limiter_override_fraction') or 0:.0%}", flush=True)
            if failures >= MAX_LIFT_FAILURES:
                raise SessionStop(f"lift failed {failures} times in a row")
            report["reset_s"] = session.reset()
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    except (Exception, KeyboardInterrupt) as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "NonFiniteFeedback" and not args.sim:
            # The firmware sends NaN after latching a safety fault; read which
            # one on this same connection (reopening USB can reset it).
            session.stop()
            try:
                faults = client.get_safety_faults()
                names = [n for bit, n in ((1, "travel"), (2, "encoder"), (4, "watchdog"), (8, "invalid command"))
                         if faults & bit]
                stop_reason += f" (firmware fault bits={faults}: {', '.join(names) or 'none latched'})"
            except (OSError, RuntimeError):
                stop_reason += " (firmware fault status unavailable)"
        print(f"Session stopped: {stop_reason}", flush=True)
    finally:
        session.stop()
        if not args.sim:
            client.__exit__(None, None, None)
            print("Session ended. Switch 12 V OFF before touching or recentering the mechanism.")
        meta.update(stop_reason=stop_reason, episodes_completed=len(reports))
        (args.output / "batch.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_batches(batches):
    data = {k: [] for k in ("obs", "next_obs", "action", "reward", "done")}
    used = []
    catch = []
    for batch in batches:
        files = sorted(glob.glob(str(Path(batch) / "transitions_*.npz")))
        meta = json.loads((Path(batch) / "batch.json").read_text())
        used.append({"batch": str(batch), "checkpoint_sha256": meta["checkpoint_sha256"],
                     "stage": meta["stage"], "deterministic": meta["deterministic"], "episodes": len(files)})
        for f in files:
            with np.load(f) as d:
                for k in data:
                    data[k].append(d[k])
                catch.append(catch_mask(d["obs"]))
    out = {k: np.concatenate(v) for k, v in data.items()}
    out["catch"] = np.concatenate(catch)
    return out, used


def guard_weights(weight, start_deg=45.0):
    """V1's reward weights with the arm-guard term (reward.py k_arm_guard) set."""
    import dataclasses
    return dataclasses.replace(reward_weights(), k_arm_guard=weight, arm_guard_start=math.radians(start_deg))


def add_guard_penalty(data, weight, start_deg=45.0):
    """Subtract the arm-guard penalty from stored rig rewards (computed with the
    term off) in place, using each transition's next observation; returns it."""
    from finetune_real import REWARD_SCALE
    from pendulum_lab.plant.reward import arm_guard_penalty
    w = guard_weights(weight, start_deg)
    nxt = data["next_obs"]
    theta = np.arctan2(nxt[:, -5], nxt[:, -4])
    penalty = np.array([arm_guard_penalty(t, a, w) for t, a in zip(theta, nxt[:, -6])])
    data["reward"] = data["reward"] - penalty * REWARD_SCALE
    return penalty


def catch_mask(obs, before_s=0.2, after_s=1.0, hz=50):
    """Ticks around each entry into the runner's balance window (|theta| <= 15 deg,
    |pendulum speed| <= 4 rad/s, newest obs frame), from before_s before entry to
    after_s after it: the catch transient, where v3b's catches are lost."""
    theta = np.arctan2(obs[:, -5], obs[:, -4])
    inside = (np.abs(theta) <= math.radians(15)) & (np.abs(obs[:, -2]) <= 4.0)
    mask = np.zeros(len(obs), dtype=bool)
    for i in np.flatnonzero(inside[1:] & ~inside[:-1]) + 1:
        mask[max(i - int(before_s * hz), 0):i + int(after_s * hz)] = True
    return mask


def update(args):
    check_lineage(args.checkpoint)
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from finetune_real import fill

    torch.manual_seed(args.seed)
    data, used = load_batches(args.batches)
    args.anchor_obs = None
    if args.anchor_min_deg is not None:
        # Swing-up states for the anchor: real observations at least this far from upright.
        far = data["obs"][:, -4] < math.cos(math.radians(args.anchor_min_deg))
        args.anchor_obs = data["obs"][far]
        print(f"anchor states (>= {args.anchor_min_deg:g} deg from upright): {int(far.sum())}", flush=True)
        used.append({"anchor_min_deg": args.anchor_min_deg, "anchor_states": int(far.sum()),
                     "anchor_steps": args.anchor_steps})
    if args.overspeed_weight is not None:
        # Re-weight the near-upright overspeed penalty in the stored rig rewards
        # so they match the simulator half (reward.py: k * gate(theta) * excess^2).
        from finetune_real import REWARD_SCALE
        w = reward_weights()
        nxt = data["next_obs"]
        theta = np.arctan2(nxt[:, -5], nxt[:, -4])
        excess = np.maximum(np.abs(nxt[:, -2]) - w.overspeed_pen_vel_limit, 0.0)
        gate = np.exp(-theta ** 2 / w.overspeed_sigma_theta ** 2)
        data["reward"] = data["reward"] - (args.overspeed_weight - w.k_upright_overspeed) * gate * excess ** 2 * REWARD_SCALE
        used.append({"overspeed_weight": args.overspeed_weight, "previous": w.k_upright_overspeed})
    if args.guard_weight is not None:
        # Add the arm-guard penalty (reward.py: k * gate(theta) * excess^2) to the
        # stored rig rewards; the simulator half gets the same weight in sim_env.
        penalty = add_guard_penalty(data, args.guard_weight, args.guard_start_deg)
        print(f"arm-guard penalty on {int((penalty > 0.01).sum())} rig ticks (mean where active "
              f"{penalty[penalty > 0.01].mean() if (penalty > 0.01).any() else 0:.2f})", flush=True)
        used.append({"guard_weight": args.guard_weight, "guard_start_deg": args.guard_start_deg,
                     "penalized_ticks": int((penalty > 0.01).sum())})
    if args.rig_max_theta_deg is not None:
        # Keep only rig ticks whose newest frame is near upright (obs frame:
        # arm, sin, cos, arm_vel, pen_vel, last_action; cos = 1 upright), so the
        # rig data shapes catching and balancing but not the swing-up.
        keep = data["obs"][:, -4] > math.cos(math.radians(args.rig_max_theta_deg))
        print(f"rig ticks within {args.rig_max_theta_deg:g} deg of upright: {int(keep.sum())} of {len(keep)}", flush=True)
        data = {k: v[keep] for k, v in data.items()}
        used.append({"rig_max_theta_deg": args.rig_max_theta_deg, "kept": int(keep.sum())})
    if args.catch_oversample > 1:
        # Repeat catch-transient ticks so they make up a larger share of the rig half.
        idx = np.flatnonzero(data["catch"])
        order = np.concatenate([np.arange(len(data["obs"]))] + [idx] * (args.catch_oversample - 1))
        print(f"catch-transient rig ticks: {len(idx)} of {len(data['obs'])}, repeated x{args.catch_oversample}", flush=True)
        data = {k: v[order] for k, v in data.items()}
        used.append({"catch_oversample": args.catch_oversample, "catch_ticks": int(len(idx))})
    if args.mirror:
        # The rig and the reward are mirror-symmetric (arm, theta and every
        # velocity and action flip sign together; every reward term is squared),
        # so each real transition also holds for the mirrored state. Adds the
        # mirrored copy of every kept rig transition with the same reward.
        flip = np.tile(MIRROR_FRAME, data["obs"].shape[1] // len(MIRROR_FRAME)).astype(data["obs"].dtype)
        mirrored = {"obs": data["obs"] * flip, "next_obs": data["next_obs"] * flip, "action": -data["action"],
                    "reward": data["reward"], "done": data["done"], "catch": data["catch"]}
        data = {k: np.concatenate([data[k], mirrored[k]]) for k in data}
        print(f"mirrored rig transitions added: {len(data['obs']) // 2}", flush=True)
        used.append({"mirror": True})
    data.pop("catch")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.sim_steps:
        return update_mixed(args, data, used)
    model = SAC.load(args.checkpoint, device="cpu", learning_rate=args.learning_rate)
    model.set_random_seed(args.seed)
    model.set_logger(configure(str(args.output / "log"), ["csv"]))
    buffer = ReplayBuffer(len(data["obs"]) + 10, model.observation_space, model.action_space,
                          device="cpu", handle_timeout_termination=False)
    fill(buffer, data)
    model.replay_buffer = buffer
    if args.critic_warmup:
        # Fit the critics to rig rewards before the actor follows them.
        frozen = [model.actor.optimizer] + ([model.ent_coef_optimizer] if model.ent_coef_optimizer else [])
        schedule = model._update_learning_rate

        def critic_only(optimizers):
            schedule(optimizers)
            for optimizer in frozen:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0
        model._update_learning_rate = critic_only
        model.train(gradient_steps=args.critic_warmup, batch_size=model.batch_size)
        model._update_learning_rate = schedule
    steps = args.gradient_steps if args.gradient_steps is not None else round(args.utd * len(data["obs"]))
    for done in range(0, steps, 500):
        model.train(gradient_steps=min(500, steps - done), batch_size=model.batch_size)
        model.logger.dump(done)
    model.save(args.output / "policy.zip")
    (args.output / "version.json").write_text(json.dumps({
        "method": "offline SAC updates (ordinary SAC losses) on on-rig policy ticks only; "
                  "no imitation, no other controllers' data",
        "parent": str(args.checkpoint), "parent_sha256": sha256(args.checkpoint),
        "batches": used, "transitions": int(len(data["obs"])), "critic_warmup": args.critic_warmup,
        "gradient_steps": steps, "learning_rate": args.learning_rate, "seed": args.seed,
        "policy_sha256": sha256(args.output / "policy.zip"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")
    print(f"{args.output / 'policy.zip'}: {steps} updates on {len(data['obs'])} rig transitions "
          f"from {len(used)} batches", flush=True)


def sim_env(seed, overspeed_weight=None, guard_weight=None, guard_start_deg=45.0):
    """V1's own training simulator (finetune_real.py as run for V1)."""
    import gymnasium as gym
    from pendulum_lab.disturbance import PendulumDisturbance
    from pendulum_lab.robustness import stress_env
    from finetune_real import REWARD_SCALE, RandomVelocityNoise
    from train_rig_rl import RIG, LimitedActions, RandomDelay
    env = stress_env(PARAMS, RIG)
    plant = env.unwrapped
    plant.exact_hanging_reset_frac = 0.35
    plant.upright_reset_frac = 0.25
    if overspeed_weight is not None:
        import dataclasses
        plant._reward_weights = dataclasses.replace(plant._reward_weights, k_upright_overspeed=overspeed_weight)
    if guard_weight is not None:
        import dataclasses
        plant._reward_weights = dataclasses.replace(plant._reward_weights, k_arm_guard=guard_weight,
                                                    arm_guard_start=math.radians(guard_start_deg))
    env = RandomDelay(env, 0.006, 0.02)
    env = PendulumDisturbance(env, 6.0, randomize=True)
    env = RandomVelocityNoise(env, 0.4, 1.2)
    env = gym.wrappers.TransformReward(LimitedActions(env), lambda r: r * REWARD_SCALE)
    env.reset(seed=seed)
    return env


def swingup_sim_buffer(min_deg):
    """MixedReplayBuffer whose simulator share is drawn only from transitions at
    least `min_deg` from upright (newest obs frame), so near upright the policy
    learns from rig transitions alone; the simulator still supplies swing-up."""
    import torch
    from stable_baselines3.common.type_aliases import ReplayBufferSamples
    from finetune_real import MixedReplayBuffer
    cos_limit = math.cos(math.radians(min_deg))

    class SwingupSimBuffer(MixedReplayBuffer):
        def sample(self, batch_size, env=None):
            n_real = int(round(batch_size * self.real_fraction)) if self.real is not None else 0
            upper = self.buffer_size if self.full else self.pos
            valid = np.flatnonzero(self.observations[:upper, 0, -4] < cos_limit)
            if len(valid) == 0:
                return super().sample(batch_size, env)
            sim = self._get_samples(np.random.choice(valid, batch_size - n_real), env)
            if n_real == 0:
                return sim
            hw = self.real.sample(n_real, env)
            return ReplayBufferSamples(*(None if a is None else torch.cat([a, b]) for a, b in zip(sim, hw)))

    return SwingupSimBuffer


def add_swingup_anchor(model, anchor_obs, steps, seed, anchor_batch=256):
    """Hold the actor's swing-up fixed: after every SAC training call, take `steps`
    extra actor steps per gradient step that pull its deterministic action on real
    swing-up states back to the parent's (the loaded checkpoint, rig-proven).
    A trust region on SAC's own previous policy; no other controller is involved."""
    import torch
    obs = torch.as_tensor(anchor_obs, dtype=torch.float32)
    with torch.no_grad():
        target = model.actor(obs, deterministic=True).clone()
    rng = np.random.default_rng(seed)
    original = model.train

    def train(gradient_steps, batch_size=64):
        original(gradient_steps=gradient_steps, batch_size=batch_size)
        model.actor.set_training_mode(True)
        losses = []
        for _ in range(gradient_steps * steps):
            idx = torch.as_tensor(rng.integers(0, len(obs), anchor_batch))
            loss = torch.nn.functional.mse_loss(model.actor(obs[idx], deterministic=True), target[idx])
            model.actor.optimizer.zero_grad()
            loss.backward()
            model.actor.optimizer.step()
            losses.append(loss.item())
        model.actor.set_training_mode(False)
        if losses:
            model.logger.record("train/anchor_loss", float(np.mean(losses)))

    model.train = train


def update_mixed(args, data, used):
    """SAC on fresh simulator experience with a fixed share of every batch drawn
    from the rig transitions (RLPD-style, as V1 itself was trained). The
    simulator half keeps swing-up and the rest of the state space anchored
    while the rig half supplies the real near-upright dynamics."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from finetune_real import MixedReplayBuffer, fill
    env = sim_env(args.seed, args.overspeed_weight, args.guard_weight, args.guard_start_deg)
    buffer_class = MixedReplayBuffer if args.sim_min_deg is None else swingup_sim_buffer(args.sim_min_deg)
    if args.sim_min_deg is not None:
        used.append({"sim_min_deg": args.sim_min_deg})
    model = SAC.load(args.checkpoint, env=env, device="cpu", learning_rate=args.learning_rate,
                     replay_buffer_class=buffer_class)
    model.set_random_seed(args.seed)
    model.set_logger(configure(str(args.output / "log"), ["csv"]))
    if args.anchor_obs is not None:
        add_swingup_anchor(model, args.anchor_obs, args.anchor_steps, args.seed)
    rig = ReplayBuffer(len(data["obs"]) + 10, model.observation_space, model.action_space,
                       device="cpu", handle_timeout_termination=False)
    fill(rig, data)
    model.replay_buffer.real, model.replay_buffer.real_fraction = rig, args.rig_fraction
    model.learning_starts = model.num_timesteps + 2000
    model.learn(total_timesteps=args.sim_steps, reset_num_timesteps=False)
    model.__dict__.pop("train", None)  # drop the anchor wrapper so it is not saved into the checkpoint
    model.replay_buffer_class = ReplayBuffer  # a locally defined buffer class cannot be pickled
    model.replay_buffer_kwargs = {}
    model.save(args.output / "policy.zip")
    (args.output / "version.json").write_text(json.dumps({
        "method": "SAC on simulator experience (V1's training simulator) with each batch "
                  f"{args.rig_fraction:.0%} on-rig policy ticks; no imitation, no other controllers' data",
        "parent": str(args.checkpoint), "parent_sha256": sha256(args.checkpoint),
        "batches": used, "transitions": int(len(data["obs"])), "sim_steps": args.sim_steps,
        "rig_fraction": args.rig_fraction, "learning_rate": args.learning_rate, "seed": args.seed,
        "policy_sha256": sha256(args.output / "policy.zip"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")
    print(f"{args.output / 'policy.zip'}: {args.sim_steps} simulator steps, {args.rig_fraction:.0%} of every "
          f"batch from {len(data['obs'])} rig transitions", flush=True)


def catch_stats(batch, hz=50):
    """Every catch (>= 0.3 s inside the runner's balance window) in a batch's
    episodes: how many are early (start < 5 s) and held >= 2 s, and held (>= 2 s)
    vs lost split by the arm's direction at entry, the asymmetry seen in e08-e11."""
    out = {"catches": 0, "early": 0, "early_held_2s": 0, "arm_neg": [0, 0], "arm_pos": [0, 0]}
    for f in sorted(glob.glob(str(Path(batch) / "episode_*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            x = z["samples"]
        if len(x) == 0:
            continue
        theta = np.angle(np.exp(1j * x[:, 2]))
        inside = np.r_[False, (np.abs(theta) <= math.radians(15)) & (np.abs(x[:, 4]) <= 4.0), False]
        edges = np.flatnonzero(np.diff(inside.astype(int)))
        for i, j in zip(edges[::2], edges[1::2]):
            if (j - i) / hz < 0.3:
                continue
            held = (j - i) / hz >= 2.0
            out["catches"] += 1
            if (x[i, 0] - x[0, 0]) < 5.0:
                out["early"] += 1
                out["early_held_2s"] += held
            out["arm_neg" if x[i, 3] < 0 else "arm_pos"][0 if held else 1] += 1
    return out


def summarize_batch(batch):
    meta = json.loads((Path(batch) / "batch.json").read_text())
    reports = json.loads((Path(batch) / "summary.json").read_text()) if (Path(batch) / "summary.json").exists() else []
    ran = [r for r in reports if r.get("samples", 0) > 0]

    def mean(key):
        values = [r[key] for r in ran if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {"batch": str(batch), "checkpoint": meta["checkpoint"], "sha": meta["checkpoint_sha256"][:10],
            "stage": meta["stage"], "deterministic": meta["deterministic"], "episodes": len(ran),
            "lift_timeouts": sum(r.get("stop_reason") == "lift_timeout" for r in reports),
            "longest_balance_s_mean": mean("longest_balance_s"),
            "longest_balance_s_median": float(np.median([r["longest_balance_s"] for r in ran])) if ran else None,
            "holds_over_1s": int(sum(r.get("holds_over_1s") or 0 for r in ran)),
            "arm_drift_deg_s_mean": mean("arm_drift_while_upright_deg_s"),
            "limiter_override_mean": mean("limiter_override_fraction"),
            "successes": int(sum(bool(r.get("success")) for r in ran)),
            "stop_reason": meta.get("stop_reason"), **catch_stats(batch)}


def report(args):
    rows = [summarize_batch(b) for b in args.batches]
    fmt = lambda v, f: "-" if v is None else format(v, f)
    print(f"{'batch':40s} {'stage':8s} det  n  lift_to  bal_mean bal_med holds drift  limiter succ  "
          f"early>=2s  arm-:held/lost arm+:held/lost")
    for r in rows:
        print(f"{Path(r['batch']).name[:40]:40s} {r['stage']:8s} {'Y' if r['deterministic'] else 'n'} "
              f"{r['episodes']:3d} {r['lift_timeouts']:5d}    {fmt(r['longest_balance_s_mean'], '6.2f')}   "
              f"{fmt(r['longest_balance_s_median'], '5.2f')} {r['holds_over_1s']:5d} "
              f"{fmt(r['arm_drift_deg_s_mean'], '5.1f')}  {fmt(r['limiter_override_mean'], '5.1%')}  "
              f"{r['successes']}/{r['episodes']}  {r['early_held_2s']:4d}/{r['early']:<4d}   "
              f"{r['arm_neg'][0]:5d}/{r['arm_neg'][1]:<5d}    {r['arm_pos'][0]:5d}/{r['arm_pos'][1]:<5d}")
    if len(rows) > 1:
        total = lambda k, i=None: sum(r[k] if i is None else r[k][i] for r in rows)
        print(f"{'all':40s} early>=2s {total('early_held_2s')}/{total('early')}  "
              f"arm- held/lost {total('arm_neg', 0)}/{total('arm_neg', 1)}  "
              f"arm+ held/lost {total('arm_pos', 0)}/{total('arm_pos', 1)}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--stage", required=True, choices=sorted(json.loads(STAGES.read_text())["stages"]))
    c.add_argument("--episodes", type=int, default=6)
    c.add_argument("--episode-seconds", type=float, default=6.0)
    c.add_argument("--deterministic", action="store_true", help="Evaluation batch: mean action, no exploration")
    c.add_argument("--seed", type=int, default=0)
    target = c.add_mutually_exclusive_group(required=True)
    target.add_argument("--port")
    target.add_argument("--sim", action="store_true")
    c.add_argument("--sim-knobs", type=Path, help="With --sim: fitted simulator knobs (fit_sim_upright.py output)")
    u = sub.add_parser("update")
    u.add_argument("--checkpoint", type=Path, required=True)
    u.add_argument("--batches", type=Path, nargs="+", required=True)
    u.add_argument("--output", type=Path, required=True)
    u.add_argument("--gradient-steps", type=int)
    u.add_argument("--utd", type=float, default=1.0, help="Gradient steps per transition when --gradient-steps is not given")
    u.add_argument("--learning-rate", type=float, default=3e-5)
    u.add_argument("--critic-warmup", type=int, default=0)
    u.add_argument("--seed", type=int, default=0)
    u.add_argument("--sim-steps", type=int, default=0,
                   help="Train on fresh simulator experience mixed with the rig data (0 = rig data only)")
    u.add_argument("--rig-fraction", type=float, default=0.5)
    u.add_argument("--overspeed-weight", type=float,
                   help="Near-upright pendulum overspeed penalty weight (default: V1's 0.02), in both halves")
    u.add_argument("--sim-min-deg", type=float,
                   help="Draw the simulator share of each batch only from transitions at least this far from upright")
    u.add_argument("--anchor-min-deg", type=float,
                   help="Hold the actor's actions fixed (to the parent's) on rig states at least this far from upright")
    u.add_argument("--anchor-steps", type=int, default=1, help="Anchor actor steps per SAC gradient step")
    u.add_argument("--catch-oversample", type=int, default=1,
                   help="Repeat rig ticks from 0.2 s before to 1 s after each balance-window entry this many times")
    u.add_argument("--rig-max-theta-deg", type=float,
                   help="Use only rig ticks within this angle of upright (default: all)")
    u.add_argument("--guard-weight", type=float,
                   help="Arm-guard penalty weight near upright (reward.py k_arm_guard), rig and simulator halves")
    u.add_argument("--guard-start-deg", type=float, default=45.0, help="Arm angle where the guard penalty starts")
    u.add_argument("--mirror", action="store_true",
                   help="Also train on the left-right mirror of every kept rig transition")
    r = sub.add_parser("report")
    r.add_argument("batches", type=Path, nargs="+")
    r.add_argument("--json", type=Path)
    args = parser.parse_args()
    {"collect": collect, "update": update, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
