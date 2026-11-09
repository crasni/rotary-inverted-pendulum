"""Reward-only SAC learning on the rig, with the sample-efficiency recipe of
real-robot RL (docs/sac-improvement-proposal.md).

One attended session, two processes (like rig_sac.py, which stays as P1's tool):

  rig loop   reset -> episode -> reset ... at 50 Hz with the runner's guards.
             Training episodes: the rig-proven LQR preset places the pendulum
             upright (its ticks are never training data), then SAC acts with
             pink (temporally correlated) exploration noise until the pendulum
             falls. Every --eval-every-th episode is the graded task instead:
             hanging start, deterministic SAC alone, 15 s.
  learner    SAC updates while the rig runs, about --utd per rig tick:
             - new critics with LayerNorm and small dropout (DroQ), fitted
               first on all earlier SAC rig data with the actor frozen;
             - each batch half this session's ticks, half earlier SAC rig data
               (RLPD symmetric sampling; no LQR logs, no other controllers);
             - the swing-up anchor of rig_curriculum.py (trust region on the
               parent's own swing-up);
             - reward with the arm-guard term (reward.py k_arm_guard).
             - several critic updates per actor update (--critic-per-actor).
             The SAC losses, automatic temperature and gamma are SAC's own.

Every episode is saved like a curriculum batch (episode_XXX.npz,
transitions_XXX.npz, summary.json, batch.json), so `rig_curriculum.py report`
works on the session. Checkpoints load in the official runner directly: the
critics inside the zip are the parent's (the runner only uses the actor); the
LayerNorm critics are saved next to them as critic_ln.pt.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path

import numpy as np

import rig_curriculum as rc
from pendulum_lab.arm_limits import limit_arm_action
from pendulum_lab.contracts import state_from_obs
from pendulum_lab.rig_session import RigSession, SimClient, save_episode
from rig_sac import PARAMS, episode_transitions, publish

PRIOR_BATCHES = "runs/curriculum/[be][0-9]*"


def pink_noise(n, rng):
    """Unit-variance 1/f (pink) Gaussian noise of length n, zero mean."""
    freqs = np.fft.rfftfreq(n)
    amplitude = np.zeros_like(freqs)
    amplitude[1:] = 1.0 / np.sqrt(freqs[1:])
    spectrum = amplitude * (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs)))
    x = np.fft.irfft(spectrum, n)
    return x / max(x.std(), 1e-8)


def layernorm_q(in_dim, hidden, dropout):
    import torch.nn as nn
    layers, width = [], in_dim
    for h in hidden:
        layers += [nn.Linear(width, h), nn.Dropout(dropout), nn.LayerNorm(h), nn.ReLU()]
        width = h
    return nn.Sequential(*layers, nn.Linear(width, 1))


def set_q_networks(critic, nets):
    critic.q_networks = list(nets)
    for i, net in enumerate(nets):
        critic.add_module(f"qf{i}", net)


class LayerNormCritics:
    """Swap SAC's critics for new DroQ-style ones (LayerNorm + dropout), and back
    to the parent's when saving, so checkpoints keep SB3's standard layout."""

    def __init__(self, model, hidden, dropout, learning_rate):
        import copy
        import torch
        self.model = model
        self.parent = (list(model.critic.q_networks), list(model.critic_target.q_networks), model.critic.optimizer)
        in_dim = model.observation_space.shape[0] + model.action_space.shape[0]
        nets = [layernorm_q(in_dim, hidden, dropout) for _ in model.critic.q_networks]
        set_q_networks(model.critic, nets)
        set_q_networks(model.critic_target, [copy.deepcopy(n) for n in nets])
        # Dropout stays on in the target critics too (DroQ): with it off, the
        # targets lose their pessimism and Q climbed past the reward's ceiling
        # (r_max / (1 - gamma) ~ 500) in a test on our rig data; with it on, Q levels off.
        model.critic_target.set_training_mode(True)
        model.critic.optimizer = torch.optim.Adam(model.critic.parameters(), lr=learning_rate)
        self.ours = (nets, list(model.critic_target.q_networks), model.critic.optimizer)

    def save(self, path):
        import torch
        m = self.model
        torch.save({"critic": [n.state_dict() for n in self.ours[0]],
                    "critic_target": [n.state_dict() for n in self.ours[1]]}, Path(path).with_suffix(".critic_ln.pt"))
        set_q_networks(m.critic, self.parent[0]); set_q_networks(m.critic_target, self.parent[1])
        m.critic.optimizer = self.parent[2]
        # The anchor wrapper and the critic-only learning-rate switch live in the
        # instance dict and reference learner-only objects: keep them out of the file.
        hooks = {k: m.__dict__.pop(k) for k in ("train", "_update_learning_rate") if k in m.__dict__}
        try:
            m.save(path)
        finally:
            m.__dict__.update(hooks)
            set_q_networks(m.critic, self.ours[0]); set_q_networks(m.critic_target, self.ours[1])
            m.critic.optimizer = self.ours[2]


def load_sac_prior(guard_weight, guard_start_deg):
    """All curriculum batches: SAC's own on-rig ticks only (every curriculum
    checkpoint has SAC-only lineage; lift ticks were never saved as transitions)."""
    batches = sorted(b for b in glob.glob(PRIOR_BATCHES) if glob.glob(f"{b}/transitions_*.npz"))
    data, _ = rc.load_batches(batches)
    data.pop("catch")
    if guard_weight:
        rc.add_guard_penalty(data, guard_weight, guard_start_deg)
    return data, batches


def learner(inbox, args):
    """Own process, lower priority, so the rig loop keeps its deadlines."""
    os.nice(10)
    import torch
    torch.set_num_threads(max(1, args.learner_threads))
    torch.manual_seed(args.seed)
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from finetune_real import MixedReplayBuffer, fill

    model = SAC.load(args.checkpoint, device="cpu", learning_rate=args.learning_rate,
                     custom_objects={"replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {}})
    model.set_random_seed(args.seed)
    model.set_logger(configure(str(args.output / "learner_log"), ["csv"]))
    critics = LayerNormCritics(model, args.critic_hidden, args.critic_dropout, args.learning_rate)
    prior, batches = load_sac_prior(args.guard_weight, args.guard_start_deg)
    obs_space, act_space = model.observation_space, model.action_space
    prior_buffer = ReplayBuffer(len(prior["obs"]) + 10, obs_space, act_space, device="cpu",
                                handle_timeout_termination=False)
    fill(prior_buffer, prior)
    online = MixedReplayBuffer(args.buffer_size, obs_space, act_space, device="cpu",
                               handle_timeout_termination=False)
    online.real, online.real_fraction = prior_buffer, args.prior_fraction
    far = prior["obs"][:, -4] < math.cos(math.radians(args.anchor_min_deg))
    rc.add_swingup_anchor(model, prior["obs"][far], args.anchor_steps, args.seed)
    print(f"[learner] prior: {len(prior['obs'])} SAC rig transitions from {len(batches)} batches; "
          f"anchor states: {int(far.sum())}", flush=True)

    # Fit the new critics to the rig data with the actor and temperature frozen.
    model.replay_buffer = prior_buffer
    frozen = [model.actor.optimizer] + ([model.ent_coef_optimizer] if model.ent_coef_optimizer else [])
    schedule = model._update_learning_rate

    def critic_only(optimizers):
        schedule(optimizers)
        for optimizer in frozen:
            for group in optimizer.param_groups:
                group["lr"] = 0.0
    model._update_learning_rate = critic_only
    t0 = time.time()
    for done in range(0, args.critic_warmup, 1000):
        SAC.train(model, gradient_steps=min(1000, args.critic_warmup - done), batch_size=model.batch_size)
    model._update_learning_rate = schedule
    model.replay_buffer = online
    print(f"[learner] critic fit: {args.critic_warmup} steps in {time.time() - t0:.0f}s; learning starts", flush=True)
    (args.output / "learner_ready").write_text("ready\n")

    collected = updates = 0
    stopping = False
    while True:
        try:
            message = inbox.get(timeout=0.02)
        except queue.Empty:
            message = None
        if message == "stop":
            stopping = True
        elif message is not None:
            index, data = message
            fill(online, data)
            collected += len(data["obs"])
        budget = int(args.utd * collected) - updates
        if budget > 0 and online.size() >= model.batch_size:
            # Many critic updates per actor update (Smith 2022: 20 to 1): the
            # critic follows the data at a high rate while the actor moves
            # about once per rig tick. Critic-only steps bypass the anchor
            # wrapper (SAC.train directly) with actor and temperature frozen.
            steps = min(budget, 100)
            actor_steps = max(1, round(steps / args.critic_per_actor))
            model._update_learning_rate = critic_only
            SAC.train(model, gradient_steps=steps - actor_steps, batch_size=model.batch_size)
            model._update_learning_rate = schedule
            model.train(gradient_steps=actor_steps, batch_size=model.batch_size)
            updates += steps
            publish(args.output, updates, model.policy.actor)
            if updates % 2000 < steps:
                model.logger.dump(updates)
        if message is not None and message != "stop":
            critics.save(args.output / "checkpoints" / f"episode_{message[0]:03d}_updates_{updates}.zip")
        if stopping and (budget <= 0 or args.no_catch_up):
            critics.save(args.output / "policy.zip")
            print(f"[learner] saved after {updates} updates on {collected} session transitions", flush=True)
            return


class Actor:
    """Current SAC actor with the shared arm limiter. Stochastic episodes draw
    SAC's Gaussian noise from a pink sequence instead of independent samples."""

    def __init__(self, checkpoint, arm_limits, noise, seed):
        from stable_baselines3 import SAC
        self.model = SAC.load(checkpoint, device="cpu", custom_objects={"replay_buffer_class": None,
                                                                         "replay_buffer_kwargs": {}})
        assert np.allclose(self.model.action_space.low, -1) and np.allclose(self.model.action_space.high, 1)
        self.arm_limits, self.noise, self.rng = arm_limits, noise, np.random.default_rng(seed)
        self.deterministic, self.updates, self.loaded_mtime = True, 0, None

    def load(self, output):
        import torch
        path = output / "actor_weights.pt"
        if not path.exists() or path.stat().st_mtime == self.loaded_mtime:
            return
        self.loaded_mtime = path.stat().st_mtime
        latest = torch.load(path, map_location="cpu")
        self.updates = latest["updates"]
        self.model.policy.actor.load_state_dict(latest["state"])

    def new_episode(self, deterministic, ticks):
        self.deterministic, self.tick = deterministic, 0
        self.sequence = pink_noise(ticks + 10, self.rng) if self.noise == "pink" else self.rng.normal(size=ticks + 10)

    def __call__(self, observation):
        import torch
        if self.deterministic:
            requested, _ = self.model.predict(observation, deterministic=True)
            requested = float(np.asarray(requested).item())
        else:
            with torch.no_grad():
                mean, log_std, _ = self.model.actor.get_action_dist_params(
                    torch.as_tensor(observation[None], dtype=torch.float32))
            eps = self.sequence[min(self.tick, len(self.sequence) - 1)]
            requested = float(torch.tanh(mean + log_std.exp() * eps).item())
        self.tick += 1
        requested = float(np.clip(requested, -1, 1))
        arm, _, velocity, _ = state_from_obs(observation)
        delay, speed, travel = self.arm_limits
        sent = limit_arm_action(requested, arm, velocity, observation[-1],
                                delay=delay, speed_target=speed, travel_target=travel)
        return sent, requested


def reached_top(episode, within_s=6.0):
    rows = np.asarray(episode.rows, dtype=float)
    if len(rows) == 0:
        return False
    first = rows[: int(within_s * 50)]
    return bool(np.any(np.abs(np.angle(np.exp(1j * first[:, 2]))) < math.radians(15)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--port")
    target.add_argument("--sim", action="store_true", help="Run the session against the simulator")
    parser.add_argument("--realtime", action="store_true", help="With --sim: real clock, to check deadlines")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--train-seconds", type=float, default=8.0, help="Training episode cap (ends 0.5 s after a fall)")
    parser.add_argument("--eval-seconds", type=float, default=15.0)
    parser.add_argument("--eval-every", type=int, default=5, help="Every Nth episode is the graded task (0 = never)")
    parser.add_argument("--stage", default="lqr_balance", help="Assisted start for training episodes (rig_curriculum.json)")
    parser.add_argument("--noise", choices=("pink", "white"), default="pink")
    parser.add_argument("--utd", type=float, default=8.0, help="SAC updates per collected rig tick")
    parser.add_argument("--prior-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--critic-dropout", type=float, default=0.01)
    parser.add_argument("--critic-warmup", type=int, default=20000, help="Critic-only steps on prior SAC rig data")
    parser.add_argument("--anchor-min-deg", type=float, default=30.0)
    parser.add_argument("--anchor-steps", type=int, default=2)
    parser.add_argument("--critic-per-actor", type=float, default=8.0,
                        help="Critic updates per actor (and temperature) update; 8 with --utd 8 = one actor step per rig tick")
    parser.add_argument("--guard-weight", type=float, default=20.0)
    parser.add_argument("--guard-start-deg", type=float, default=45.0)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learner-threads", type=int, default=4)
    parser.add_argument("--no-catch-up", action="store_true", help="On stop, save without finishing owed updates")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rc.check_lineage(args.checkpoint)
    limits = rc.arm_limits()
    stage = json.loads(rc.STAGES.read_text())["stages"][args.stage]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()
    meta = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": rc.sha256(args.checkpoint),
            "stage": "online", "training_stage": args.stage, "stage_parameters": stage, "deterministic": False,
            "method": "on-rig SAC: LayerNorm/dropout critics, symmetric sampling with earlier SAC rig data, "
                      "swing-up anchor, pink exploration noise, arm-guard reward; no imitation, no other "
                      "controllers' data",
            "arguments": {k: str(v) for k, v in vars(args).items()}, "arm_limits": limits,
            "target": "sim" if args.sim else args.port, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (args.output / "batch.json").write_text(json.dumps(meta, indent=2) + "\n")

    ctx = mp.get_context("spawn")
    inbox = ctx.Queue()
    learner_process = ctx.Process(target=learner, args=(inbox, args))
    learner_process.start()
    actor = Actor(args.checkpoint, limits, args.noise, args.seed)
    weights = rc.guard_weights(args.guard_weight, args.guard_start_deg) if args.guard_weight else rc.reward_weights()
    rng = np.random.default_rng(args.seed)
    lifter = rc.lqr_lifter()

    episode_ticks = {True: round(args.eval_seconds * 50), False: round(args.train_seconds * 50)}
    seconds = max(args.eval_seconds, args.train_seconds)
    if args.sim:
        client = SimClient(PARAMS, seed=args.seed)
        if args.realtime:
            session = RigSession(client, arm_limits=limits, episode_seconds=seconds)
        else:
            clock = [0.0]
            session = RigSession(client, arm_limits=limits, episode_seconds=seconds, clock=lambda: clock[0],
                                 sleep=lambda d: clock.__setitem__(0, clock[0] + max(d, 0)), log=lambda *_: None)
        confirm = lambda _: "RUN"
    else:
        from pendulum_lab.serial_client import LowLevelClient
        client = LowLevelClient(args.port, timeout=0.1).__enter__()
        session = RigSession(client, arm_limits=limits, episode_seconds=seconds)
        confirm = input

    reports, stop_reason, lift_failures, broken_swingups = [], "completed", 0, 0
    try:
        print("Fitting the new critics before the first update (about a minute); "
              "the first episodes run the parent policy meanwhile.", flush=True)
        session.start(confirm=confirm)
        for index in range(args.episodes):
            actor.load(args.output)
            evaluation = bool(args.eval_every) and index % args.eval_every == args.eval_every - 1
            session.episode_ticks = episode_ticks[evaluation]
            actor.new_episode(evaluation, session.episode_ticks)
            start = None if evaluation else rc.start_for(stage, rng, lifter)
            episode = session.run_episode(actor, start=start)
            report = save_episode(args.output, index, episode, {
                "episode": index, "deterministic": evaluation, "assisted": start is not None,
                "policy_updates": actor.updates, "noise": None if evaluation else args.noise})
            if len(episode.rows) >= 3:
                data = episode_transitions(episode, weights)
                np.savez_compressed(args.output / f"transitions_{index:03d}.npz", **data)
                inbox.put((index, data))
            reports.append(report)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            if episode.stop_reason == "lift_timeout":
                lift_failures += 1
                print(f"episode {index}: LQR placement did not reach the window in time", flush=True)
                if lift_failures >= rc.MAX_LIFT_FAILURES:
                    raise rc.SessionStop(f"placement failed {lift_failures} times in a row")
            else:
                lift_failures = 0
                kind = "EVAL from hanging, deterministic" if evaluation else f"train ({args.noise} noise)"
                print(f"episode {index} [{kind}] after {actor.updates} updates: "
                      f"longest balance {report['longest_balance_s']:.2f}s, success {report['success']}, "
                      f"limiter {report.get('limiter_override_fraction') or 0:.0%}", flush=True)
            if evaluation:
                broken_swingups = 0 if reached_top(episode) else broken_swingups + 1
                if broken_swingups >= 2:
                    raise rc.SessionStop("swing-up failed in two evaluations in a row")
            report["reset_s"] = session.reset()
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    except (Exception, KeyboardInterrupt) as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
        print(f"Session stopped: {stop_reason}", flush=True)
    finally:
        session.stop()
        if not args.sim:
            client.__exit__(None, None, None)
            print("Session ended. Switch 12 V OFF before touching or recentering the mechanism.")
        print("Saving the final checkpoint...", flush=True)
        inbox.put("stop")
        learner_process.join(timeout=900)
        meta.update(stop_reason=stop_reason, episodes_completed=len(reports))
        (args.output / "batch.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
