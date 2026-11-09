"""SAC fine-tuning on the real rig, with automatic resets between episodes.

The simulator does not predict how new policies behave near upright on this
rig (three validated simulator changes each failed on a new policy), so this
tool lets SAC learn from the rig itself. The policy that results is still a
plain SAC policy that swings up and balances on its own.

Two processes:
  rig loop   (this process) one attended session: reset -> episode -> reset ...
             at 50 Hz with the hardware runner's guards (pendulum_lab.rig_session).
             Acts with the current stochastic SAC actor (--learn) or the
             deterministic policy (--evaluate).
  learner    SAC updates between and during episodes, RLPD-style: each batch is
             half on-rig transitions from this session and half earlier rig logs
             (all algorithms, including successful LQR balancing). Updates are
             capped at --utd per collected transition. New actor weights reach
             the rig loop at each episode start; a checkpoint is saved per episode.

Every episode is saved like a hardware run (episode_XXX.npz, summary.json) so the
usual analysis applies. Use --sim to run the whole session against the simulator.
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

from pendulum_lab.arm_limits import limit_arm_action
from pendulum_lab.contracts import PROFILE, state_from_obs
from pendulum_lab.rig_session import RigSession, SessionStop, SimClient, save_episode

PARAMS = Path("configs/my_rig.json")
SAC_CONFIG = Path("configs/sac_my_rig.json")
REWARD_SCALE = 0.2


def reward_weights():
    from pendulum_lab.robustness import stress_env
    from train_rig_rl import RIG
    return stress_env(PARAMS, RIG).unwrapped._reward_weights


def episode_transitions(episode, weights):
    """(obs, next_obs, policy action, reward, done) from one on-rig episode."""
    from pendulum_lab.plant.reward import compute_reward
    rows = np.asarray(episode.rows, dtype=float)
    obs = np.asarray(episode.observations, dtype=np.float32)
    actions = np.asarray(episode.actions, dtype=np.float32)
    out = {k: [] for k in ("obs", "next_obs", "action", "reward", "done")}
    for k in range(1, len(rows) - 1):
        arm, theta, arm_vel, pen_vel = rows[k + 1, 1:5]
        r = compute_reward(theta=theta, pen_vel=pen_vel, motor_pos=arm, motor_vel=arm_vel,
                           action=float(rows[k, 5]), prev_action=float(rows[k - 1, 5]),
                           prev_motor_vel=float(rows[k, 3]), weights=weights)
        out["obs"].append(obs[k]); out["next_obs"].append(obs[k + 1])
        out["action"].append([float(np.clip(actions[k], -1, 1))])
        out["reward"].append(r * REWARD_SCALE); out["done"].append(0.0)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def prior_runs():
    runs = glob.glob("docs/results/hardware/*/2026*") + glob.glob("runs/*_rig_2026*") \
        + glob.glob("runs/sac_*_2026*") + glob.glob("runs/custom_*_2026*")
    keep = []
    for r in sorted(set(runs)):
        if "163412" in r or not os.path.exists(f"{r}/summary.json") or not os.path.exists(f"{r}/episode_000.npz"):
            continue
        try:
            if json.loads(Path(f"{r}/summary.json").read_text())[0].get("samples", 0) >= 20:
                keep.append(r)
        except (ValueError, IndexError, KeyError):
            continue
    return keep


PRIOR_CACHE = Path("runs/rig_sac_prior.npz")


def load_prior():
    """Earlier rig logs as transitions, cached by the list of runs they came from."""
    from finetune_real import load_real
    runs = prior_runs()
    if PRIOR_CACHE.exists():
        cached = np.load(PRIOR_CACHE, allow_pickle=False)
        if list(cached["runs"]) == runs:
            return {k: cached[k] for k in ("obs", "next_obs", "action", "reward", "done")}
    prior = load_real(runs, reward_weights())
    np.savez_compressed(PRIOR_CACHE, runs=np.asarray(runs), **prior)
    return prior


def publish(output, updates, actor):
    """Atomically replace the actor weights file read by the rig loop."""
    import torch
    tmp = output / "actor_weights.tmp"
    torch.save({"updates": updates, "state": actor.state_dict()}, tmp)
    os.replace(tmp, output / "actor_weights.pt")


def learner(checkpoint, output, inbox, args):
    """Runs in its own process at lower priority so the rig loop keeps its deadlines."""
    os.nice(10)
    import torch
    torch.set_num_threads(max(1, args.learner_threads))
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from finetune_real import MixedReplayBuffer, fill

    model = SAC.load(checkpoint, device="cpu", learning_rate=args.learning_rate)
    model.set_logger(configure(str(output / "learner_log"), ["csv"]))
    obs_space, act_space = model.observation_space, model.action_space
    online = MixedReplayBuffer(args.buffer_size, obs_space, act_space, device="cpu",
                               handle_timeout_termination=False)
    prior = load_prior()
    prior_buffer = ReplayBuffer(len(prior["obs"]) + 10, obs_space, act_space, device="cpu",
                                handle_timeout_termination=False)
    fill(prior_buffer, prior)
    online.real, online.real_fraction = prior_buffer, args.prior_fraction
    print(f"[learner] prior rig transitions: {len(prior['obs'])}", flush=True)
    if args.critic_warmup:
        # The checkpoint's critic was trained for the policy before any
        # pretraining; fit it to the rig data with the actor and entropy
        # frozen, so the first actor updates are not pulled back toward the
        # old behaviour.
        model.replay_buffer = prior_buffer
        frozen = [model.actor.optimizer]
        if model.ent_coef_optimizer is not None:
            frozen.append(model.ent_coef_optimizer)
        schedule = model._update_learning_rate
        def critic_only(optimizers):
            schedule(optimizers)
            for optimizer in frozen:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0
        model._update_learning_rate = critic_only
        model.train(gradient_steps=args.critic_warmup, batch_size=model.batch_size)
        model._update_learning_rate = schedule
        print(f"[learner] critic warmup: {args.critic_warmup} steps on prior rig data", flush=True)
    model.replay_buffer = online

    bc = None
    if args.bc_ratio > 0:
        # Keep the actor near LQR's successful rig balancing while SAC adjusts
        # it: alternate SAC chunks with imitation steps on the demonstrations.
        from pretrain_actor import demonstrations, lqr_runs, mean_action
        demo_obs, demo_act = demonstrations(lqr_runs())
        bc = (torch.as_tensor(demo_obs), torch.as_tensor(demo_act),
              torch.optim.Adam(model.actor.parameters(), lr=args.bc_learning_rate), np.random.default_rng(0))
        print(f"[learner] behaviour cloning on {len(demo_obs)} LQR balancing ticks, ratio {args.bc_ratio}", flush=True)

    collected = updates = episodes = 0
    stopping = False
    while True:
        try:
            message = inbox.get(timeout=0.05)
        except queue.Empty:
            message = None
        if message == "stop":
            stopping = True
        elif message is not None:
            index, data = message
            fill(online, data)
            collected += len(data["obs"]); episodes += 1
        budget = int(args.utd * collected) - updates
        if episodes >= args.warmup_episodes and budget > 0:
            steps = min(budget, 50)
            model.train(gradient_steps=steps, batch_size=model.batch_size)
            if bc is not None:
                demo_obs, demo_act, bc_optimizer, bc_rng = bc
                for _ in range(max(1, round(steps * args.bc_ratio))):
                    i = torch.as_tensor(bc_rng.integers(0, len(demo_obs), 256))
                    loss = torch.mean((mean_action(model.actor, demo_obs[i]) - demo_act[i]) ** 2)
                    bc_optimizer.zero_grad(); loss.backward(); bc_optimizer.step()
            updates += steps
            publish(output, updates, model.policy.actor)
        if message is not None and message != "stop":
            model.save(output / "checkpoints" / f"episode_{message[0]:03d}_updates_{updates}")
        if stopping and (budget <= 0 or episodes < args.warmup_episodes):
            model.save(output / "policy.zip")
            print(f"[learner] saved after {updates} updates on {collected} rig transitions", flush=True)
            return


class Actor:
    """Current SAC actor on CPU with the shared arm limiter."""

    def __init__(self, checkpoint, arm_limits, deterministic):
        from stable_baselines3 import SAC
        self.model = SAC.load(checkpoint, device="cpu")
        self.arm_limits = arm_limits
        self.deterministic = deterministic
        self.updates = 0
        self.loaded_mtime = None

    def load(self, output):
        """Take the learner's newest actor weights, if any, at an episode start."""
        import torch
        path = output / "actor_weights.pt"
        if not path.exists() or path.stat().st_mtime == self.loaded_mtime:
            return
        self.loaded_mtime = path.stat().st_mtime
        latest = torch.load(path, map_location="cpu")
        self.updates = latest["updates"]
        self.model.policy.actor.load_state_dict(latest["state"])

    def __call__(self, observation):
        requested, _ = self.model.predict(observation, deterministic=self.deterministic)
        requested = float(np.clip(np.asarray(requested).item(), -1, 1))
        arm, _, velocity, _ = state_from_obs(observation)
        delay, speed, travel = self.arm_limits
        sent = limit_arm_action(requested, arm, velocity, observation[-1],
                                delay=delay, speed_target=speed, travel_target=travel)
        return sent, requested


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--learn", action="store_true", help="Stochastic SAC with on-rig updates")
    mode.add_argument("--evaluate", action="store_true", help="Deterministic policy, no updates")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--port")
    target.add_argument("--sim", action="store_true", help="Run against the simulator (testing)")
    parser.add_argument("--realtime", action="store_true",
                        help="With --sim: real clock and sleeps, to check deadlines while the learner runs")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=12.0)
    parser.add_argument("--alternate", action="store_true",
                        help="With --evaluate: alternate deterministic and sampled-action episodes")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="With --learn: every Nth episode runs the deterministic policy (0 = never)")
    parser.add_argument("--utd", type=float, default=1.0, help="Gradient steps per rig transition")
    parser.add_argument("--prior-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-episodes", type=int, default=1)
    parser.add_argument("--bc-ratio", type=float, default=0.0,
                        help="Imitation steps on LQR balancing demonstrations per SAC update (0 = off)")
    parser.add_argument("--bc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-warmup", type=int, default=3000,
                        help="Critic-only steps on prior rig data before any actor update")
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learner-threads", type=int, default=2)
    args = parser.parse_args()

    config = json.loads(SAC_CONFIG.read_text())["arm_command_limits"]
    arm_limits = (config["delay_s"], config["speed_rad_s"], config["travel_rad"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()
    (args.output / "session.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                          "arm_limits": arm_limits}, indent=2) + "\n")

    actor = Actor(args.checkpoint, arm_limits, deterministic=args.evaluate)
    ctx = mp.get_context("spawn")
    inbox = ctx.Queue()
    learner_process = None
    if args.learn:
        learner_process = ctx.Process(target=learner, args=(args.checkpoint, args.output, inbox, args))
        learner_process.start()
    weights = reward_weights()

    if args.sim and args.realtime:
        client = SimClient(PARAMS)
        session = RigSession(client, arm_limits=arm_limits, episode_seconds=args.episode_seconds)
        confirm = lambda _: "RUN"
    elif args.sim:
        client = SimClient(PARAMS)
        clock = [0.0]
        session = RigSession(client, arm_limits=arm_limits, episode_seconds=args.episode_seconds,
                             clock=lambda: clock[0], sleep=lambda d: clock.__setitem__(0, clock[0] + max(d, 0)))
        confirm = lambda _: "RUN"
    else:
        from pendulum_lab.serial_client import LowLevelClient
        client = LowLevelClient(args.port, timeout=0.1).__enter__()
        session = RigSession(client, arm_limits=arm_limits, episode_seconds=args.episode_seconds)
        confirm = input

    reports = []
    stop_reason = "completed"
    try:
        session.start(confirm=confirm)
        for index in range(args.episodes):
            actor.load(args.output)
            if args.evaluate:
                evaluation = not (args.alternate and index % 2 == 1)
            else:
                evaluation = bool(args.eval_every) and index % args.eval_every == args.eval_every - 1
            actor.deterministic = bool(evaluation)
            episode = session.run_episode(actor)
            report = save_episode(args.output, index, episode,
                                  {"episode": index, "policy_updates": actor.updates,
                                   "deterministic": bool(evaluation)})
            reports.append(report)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(f"episode {index}: longest balance {report['longest_balance_s']:.2f}s, "
                  f"balanced {report['balanced_fraction']:.0%}, success {report['success']}, "
                  f"limiter {report['limiter_override_fraction'] or 0:.0%}, "
                  f"policy after {actor.updates} updates{' [deterministic]' if evaluation else ''}", flush=True)
            if args.learn:
                inbox.put((index, episode_transitions(episode, weights)))
            reset_s = session.reset()
            reports[-1]["reset_s"] = reset_s
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(f"  reset {reset_s:.1f}s", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
        print(f"Session stopped: {stop_reason}", flush=True)
    finally:
        session.stop()
        if not args.sim:
            client.__exit__(None, None, None)
            print("Session ended. Switch 12 V OFF before touching or recentering the mechanism.")
        if learner_process is not None:
            inbox.put("stop")
            learner_process.join(timeout=600)
        (args.output / "session.json").write_text(json.dumps(
            {**json.loads((args.output / "session.json").read_text()), "stop_reason": stop_reason,
             "episodes_completed": len(reports)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
