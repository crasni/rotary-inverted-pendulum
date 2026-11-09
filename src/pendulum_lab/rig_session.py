"""Attended on-rig session: episodes separated by automatic resets.

One USB connection and one engagement for the whole session, so the step
count stays valid and the firmware watchdog is never left without commands.
Every tick uses the same guards and deadlines as `hardware.run`; any guard,
deadline miss, firmware fault or failed reset ends the session, disengages
the motor and needs a new attended start (never an automatic re-engage).

Between episodes `ResetController` damps the pendulum and recentres the arm in
acceleration mode, then a position move puts the step count exactly on zero
(it holds there without the acceleration watchdog), and the runner's own
start condition is checked before the next episode.

Assisted starts (training curriculum only): after that same start check,
`LiftController` swings the pendulum up under the same per-tick guards until
the state enters a near-upright window, and the policy takes over on that
tick. Nobody touches the mechanism while the motor is engaged. Lift ticks are
recorded separately and are never policy data.
"""
from collections import deque
from dataclasses import dataclass, field
import json
import math
import time

import numpy as np

from .arm_limits import limit_arm_action
from .contracts import PROFILE, checked_action, observation_frame, physical_state, wrap
from .evaluation import COLUMNS, summarize
from .hardware import TIMING_COLUMNS, NonFiniteFeedback, preflight, validate_state
from .rig_reset import LiftController, ResetController

RESET_TIMEOUT_S = 40.0
SNAP_TIMEOUT_S = 5.0
HANGING_RAD_S = 12.27   # small-swing natural frequency of this pendulum (configs/my_rig.json)
NEAR_ARM_RAD = math.radians(6.0)
NEAR_ARM_SPEED = 0.4
NEAR_SWING_RAD = math.radians(1.5)
NEAR_TICKS = 15
# The runner's start check allows 5 deg and 0.5 rad/s; at 12.27 rad/s a swing
# that never exceeds 0.5 rad/s has an amplitude below 2.3 deg.
SETTLED_SWING_RAD = math.radians(1.8)


def swing_amplitude(x):
    """Amplitude of the small hanging swing, from angle and speed together.

    Angle or speed alone can both be small at one instant of a larger swing.
    """
    return math.hypot(wrap(x[1] - math.pi), x[3] / HANGING_RAD_S)


class SessionStop(RuntimeError):
    """The session cannot continue safely; the motor has been disengaged."""


@dataclass
class Episode:
    rows: list
    timing: list
    observations: list   # observation before each action
    actions: list        # the policy's own (pre-limit) action
    stop_reason: str
    lift_rows: list = field(default_factory=list)   # [t, *x, sent] while lifting
    handover: list = None                           # state on the first policy tick


@dataclass
class AssistedStart:
    """Lift to `target_energy`, hand over inside the window (all magnitudes)."""
    target_energy: float
    theta_rad: float
    pen_speed: float
    arm_rad: float = math.radians(20.0)
    arm_speed: float = 1.5
    timeout_s: float = 20.0
    tolerance: float = 0.06
    lifter: object = None        # controller used instead of LiftController (e.g. rig-proven LQR)
    hold_s: float = 0.0          # with a lifter that balances: hand over after it held this long
    end_after_fall_s: float = None   # end the episode this long after |theta| > 45 deg

    def inside(self, x):
        return (abs(wrap(x[1])) < self.theta_rad and abs(x[3]) < self.pen_speed
                and abs(x[0]) < self.arm_rad and abs(x[2]) < self.arm_speed)


class Ticker:
    """50 Hz schedule with the hardware runner's deadline checks."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.start = self.clock()
        self.tick = 0

    def wait(self):
        self.tick += 1
        deadline = self.start + self.tick * PROFILE.dt
        self.sleep(max(0.0, deadline - self.clock()))
        return deadline

    def check(self, deadline, what):
        if self.clock() - deadline > 0.04:
            raise RuntimeError(f"{what} deadline missed")


class RigSession:
    def __init__(self, client, *, arm_limits, episode_seconds, clock=time.monotonic,
                 sleep=time.sleep, log=print):
        if not 0 < episode_seconds <= 30:
            raise ValueError("Episode duration must be in (0, 30] s")
        self.client = client
        self.arm_limits = arm_limits
        self.episode_ticks = round(episode_seconds * PROFILE.hz)
        self.reset_controller = ResetController(arm_limits=arm_limits)
        self.clock, self.sleep, self.log = clock, sleep, log
        self.previous_time_us = None
        self.last_action = 0.0
        self.engaged = False

    # -- lifecycle --------------------------------------------------------
    def start(self, confirm=input):
        preflight(self.client)
        if confirm("Preflight passed. Power ON, keep clear, type RUN: ").strip() != "RUN":
            raise RuntimeError("Powered session cancelled")
        s = self.client.get_state()
        validate_state(s, startup=True)
        self.previous_time_us = s.time_us
        self.client.engage_motor()
        self.engaged = True

    def stop(self):
        if self.engaged:
            self.client.disengage_motor()
            self.engaged = False

    # -- one guarded tick -------------------------------------------------
    def read(self, ticker, deadline):
        s = self.client.get_state()
        x = validate_state(s, self.previous_time_us)
        self.previous_time_us = s.time_us
        ticker.check(deadline, "Feedback")
        return s, x

    def write_acceleration(self, ticker, deadline, action):
        ticker.check(deadline, "Controller")
        if not self.client.get_motor_status().motor_engaged:
            raise RuntimeError("Firmware motor disabled or watchdog tripped")
        ticker.check(deadline, "Command")
        self.client.set_acceleration(action * PROFILE.max_accel)
        self.last_action = action

    # -- episode ----------------------------------------------------------
    def run_episode(self, policy, start=None):
        """policy(observation) -> (sent action, requested action).

        Every episode begins from the runner's hanging start check. With an
        `AssistedStart`, the lift runs first and the episode's policy ticks
        begin when the state enters its window; if it does not within the
        timeout, the episode has no policy ticks and stop_reason "lift_timeout".
        """
        s = self.client.get_state()
        x = validate_state(s, self.previous_time_us, startup=True)
        self.previous_time_us = s.time_us
        self.last_action = 0.0
        history = deque([observation_frame(x, 0.0)] * PROFILE.history, maxlen=PROFILE.history)
        episode = Episode([], [], [], [], "duration")
        ticker = Ticker(self.clock, self.sleep)
        lift, held, fallen = None, 0, 0
        if start is not None and start.lifter is not None:
            lift = start.lifter
            lift.reset()
        elif start is not None:
            lift = LiftController(arm_limits=self.arm_limits, target_energy=start.target_energy,
                                  tolerance=start.tolerance)
        while len(episode.rows) < self.episode_ticks:
            deadline = ticker.wait()
            read_start = ticker.clock() - ticker.start
            s, x = self.read(ticker, deadline)
            read_end = ticker.clock() - ticker.start
            history.append(observation_frame(x, self.last_action))
            if lift is not None:
                if start.lifter is not None:
                    held = held + 1 if getattr(lift, "balancing", False) else 0
                    ready = held * PROFILE.dt >= start.hold_s
                else:
                    ready = True
                if ready and start.inside(x):
                    lift = None
                    episode.handover = [float(v) for v in x]
                else:
                    if ticker.tick * PROFILE.dt > start.timeout_s:
                        episode.stop_reason = "lift_timeout"
                        return episode
                    if start.lifter is not None:
                        set_time = getattr(lift, "set_sample_time_us", None)
                        if callable(set_time):
                            set_time(s.time_us)
                        action = lift.act(np.concatenate(history))
                    else:
                        action = lift.act(x, self.last_action)
                    self.write_acceleration(ticker, deadline, checked_action(action))
                    episode.lift_rows.append([ticker.clock() - ticker.start, *x, action])
                    continue
            observation = np.concatenate(history)
            sent, requested = policy(observation)
            sent = checked_action(sent)
            write_start = ticker.clock() - ticker.start
            self.write_acceleration(ticker, deadline, sent)
            write_end = ticker.clock() - ticker.start
            episode.rows.append([write_end, *x, sent])
            episode.timing.append((len(episode.rows) - 1, s.time_us, read_start, read_end,
                                   write_start, write_end, -1, requested, requested,
                                   math.nan, math.nan))
            episode.observations.append(observation)
            episode.actions.append(requested)
            if start is not None and start.end_after_fall_s is not None:
                # Count from the first time past 45 deg; a spinning pendulum
                # passes upright again, which must not restart the count.
                if fallen or abs(wrap(x[1])) > math.radians(45):
                    fallen += 1
                if fallen * PROFILE.dt >= start.end_after_fall_s:
                    episode.stop_reason = "fell"
                    break
        return episode

    # -- reset ------------------------------------------------------------
    def reset(self):
        """Damp and recentre, snap the step count to zero, then check the start condition."""
        rc = self.reset_controller
        rc.reset()
        ticker = Ticker(self.clock, self.sleep)
        near = 0
        while True:
            deadline = ticker.wait()
            _, x = self.read(ticker, deadline)
            near = near + 1 if (abs(x[0]) < NEAR_ARM_RAD and abs(x[2]) < NEAR_ARM_SPEED
                                and swing_amplitude(x) < NEAR_SWING_RAD) else 0
            if near >= NEAR_TICKS:
                break
            if ticker.tick * PROFILE.dt > RESET_TIMEOUT_S:
                raise SessionStop("Reset did not settle the pendulum in time")
            self.write_acceleration(ticker, deadline, rc.act(x, self.last_action))
        # Position move to exactly zero steps; holds without the acceleration watchdog.
        self.client.set_target(0.0)
        self.last_action = 0.0
        snap = Ticker(self.clock, self.sleep)
        still = 0
        while True:
            deadline = snap.wait()
            _, x = self.read(snap, deadline)
            status = self.client.get_motor_status()
            if not status.motor_engaged:
                raise SessionStop("Motor disengaged during reset")
            settled = (not status.planner_running and abs(x[0]) < math.radians(1.0)
                       and abs(x[2]) < 0.2 and swing_amplitude(x) < SETTLED_SWING_RAD)
            still = still + 1 if settled else 0
            if still >= 10:
                return snap.tick * PROFILE.dt + ticker.tick * PROFILE.dt
            if snap.tick * PROFILE.dt > SNAP_TIMEOUT_S:
                raise SessionStop("Arm did not settle on zero after the position move")


def episode_metrics(rows, requested):
    """Progress measures beyond pass/fail: how long balance lasts, how far the
    arm drifts while upright, and how often the arm limiter overrides the policy."""
    if len(rows) == 0:
        return {}
    arm, theta, pen_speed, sent = rows[:, 1], rows[:, 2], rows[:, 4], rows[:, 5]
    upright = (np.abs(theta) <= math.radians(15)) & (np.abs(pen_speed) <= 4)
    stretches, start = [], None
    for k, value in enumerate(np.r_[upright, False]):
        if value and start is None:
            start = k
        elif not value and start is not None:
            stretches.append((start, k)); start = None
    held = [(a, b) for a, b in stretches if b - a >= PROFILE.hz // 2]
    drift = [abs(arm[b - 1] - arm[a]) / ((b - a) / PROFILE.hz) for a, b in held]
    return {
        "holds_over_1s": int(sum(b - a >= PROFILE.hz for a, b in stretches)),
        "arm_drift_while_upright_deg_s": float(np.degrees(np.mean(drift))) if drift else None,
        "max_arm_while_upright_deg": float(np.degrees(np.abs(arm[upright]).max())) if upright.any() else None,
        "limiter_override_fraction": float(np.mean(np.abs(requested - sent) > 0.2)) if len(requested) == len(sent) else None,
    }


def save_episode(directory, index, episode, extra=None):
    rows = np.asarray(episode.rows, dtype=float).reshape(-1, len(COLUMNS))
    np.savez_compressed(directory / f"episode_{index:03d}.npz", samples=rows, columns=COLUMNS,
                        hardware_timing=np.asarray(episode.timing, dtype=float).reshape(-1, len(TIMING_COLUMNS)),
                        hardware_timing_columns=TIMING_COLUMNS,
                        observations=np.asarray(episode.observations, dtype=np.float32),
                        policy_actions=np.asarray(episode.actions, dtype=np.float32))
    lift = np.asarray(episode.lift_rows, dtype=float).reshape(-1, len(COLUMNS))
    if len(lift) or episode.handover is not None:
        with np.load(directory / f"episode_{index:03d}.npz") as saved:
            arrays = dict(saved)
        np.savez_compressed(directory / f"episode_{index:03d}.npz", **arrays, lift_samples=lift,
                            handover=np.asarray(episode.handover if episode.handover else [], dtype=float))
    if len(rows) == 0:
        result = {"stop_reason": episode.stop_reason, "samples": 0, "success": False, "longest_balance_s": 0.0}
    else:
        result = summarize(rows, episode.stop_reason, len(rows) / PROFILE.hz)
        result.update(episode_metrics(rows, np.asarray(episode.actions, dtype=float)))
    result["lift_s"] = len(lift) / PROFILE.hz
    result["handover"] = episode.handover
    if extra:
        result.update(extra)
    return result


class SimClient:
    """LowLevelClient stand-in backed by the rig simulator, advanced one control
    period per get_state call, for testing sessions without hardware."""

    def __init__(self, params, *, arm_lag_s=0.018, kick=0.25, delay_s=0.018, seed=0,
                 coupling=1.0, friction=1.0, gravity=1.0):
        from .plant.pendulum_env import RotaryInvertedPendulumEnv
        self.env = RotaryInvertedPendulumEnv(
            params_path=params, control_freq_hz=PROFILE.hz, action_mode="accel",
            action_smooth_window=1, max_accel_rad_s2=PROFILE.max_accel,
            motor_max_accel_rad_s2=PROFILE.max_accel, transport_delay_s=delay_s,
            motor_planner_speed_limit_rad_s=4.4, firmware_obs_model=True,
            episode_length_s=3600.0, terminate_on_spin_loop=False)
        self.env.exact_hanging_reset_frac = 1.0
        self.env.reset(seed=seed)
        self.env._arm_lag_s = arm_lag_s
        self.env._accel_transport.zero_cross_kick_rad_s = kick
        # Optional fitted near-upright knobs (scripts/fit_sim_upright.py); 1.0 = unchanged.
        self.env._coupling_gain = coupling
        if friction != 1.0:
            import dataclasses
            e = self.env
            e.params = dataclasses.replace(e.params, pendulum_friction=e.params.pendulum_friction * friction,
                                           pendulum_coulomb=e.params.pendulum_coulomb * friction)
            e.model.dof_damping[e._pen_dof_addr] = e.params.pendulum_friction
            e.model.dof_frictionloss[e._pen_dof_addr] = e.params.pendulum_coulomb
        if gravity != 1.0:
            self.env.model.opt.gravity[:] = [0.0, 0.0, -9.81 * gravity]
        self.env.seed_measurement_ring()
        self.engaged = False
        self.accel = 0.0
        self.target = None
        self.time_us = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def wait_until_ready(self, **_):
        return True

    def require_profile(self, **_):
        return None

    def get_safety_faults(self):
        return 0

    def _step(self):
        e = self.env
        if self.target is not None:
            # moveTo(): time-optimal ramp at 49 rad/s^2 that stops exactly on
            # the target step, applied without the acceleration-command delay.
            planner = e._accel_transport
            x, v = planner.position_rad, planner.velocity_rad_s
            for _ in range(20):  # 1 ms kinematic profile: brake exactly onto the target
                error = self.target - x
                if abs(error) < 1e-4 and abs(v) < 0.05:
                    x, v = self.target, 0.0
                    continue
                wanted = math.copysign(min(4.0, math.sqrt(2 * 49.0 * abs(error))), error)
                v += float(np.clip((wanted - v) / 1e-3, -49.0, 49.0)) * 1e-3
                x += v * 1e-3
            accel = (v - planner.velocity_rad_s) / PROFILE.dt
            # A planned moveTo() ramp has no acceleration-command delay or
            # zero-crossing kick; the physical arm still lags the step count.
            saved = planner.delay_s, planner.zero_cross_kick_rad_s
            planner.delay_s, planner.zero_cross_kick_rad_s = 0.0, 0.0
            planner.reset(position_rad=planner.position_rad, velocity_rad_s=planner.velocity_rad_s,
                          active_accel_rad_s2=accel)
            try:
                e.step(np.array([accel / PROFILE.max_accel], dtype=np.float32))
            finally:
                planner.delay_s, planner.zero_cross_kick_rad_s = saved
            planner.position_rad, planner.velocity_rad_s = x, v
            planner.active_accel_rad_s2 = 0.0
        else:
            accel = self.accel if self.engaged else 0.0
            e.step(np.array([accel / PROFILE.max_accel], dtype=np.float32))
        self.time_us += 20000

    def get_state(self):
        from .serial_client import State
        self._step()
        e = self.env
        return State(time_us=self.time_us, motor_pos_rad=e._meas_motor_pos,
                     pendulum_pos_rad=wrap(e._meas_pen_phi), motor_vel_rad_s=e._meas_motor_vel,
                     pendulum_vel_rad_s=e._meas_pen_vel)

    def get_motor_status(self):
        from .serial_client import MotorStatus
        e = self.env
        running = self.target is not None and (abs(self.target - e._meas_motor_pos) > 2e-3
                                               or abs(e._meas_motor_vel) > 0.05)
        return MotorStatus(3, 0, 0, 0, 0, 0, self.engaged, running)

    def engage_motor(self):
        self.engaged = True

    def disengage_motor(self):
        self.engaged = False

    def set_acceleration(self, accel):
        if abs(accel) > 50.0:
            raise ValueError("invalid command")
        self.target = None
        self.accel = float(accel)

    def set_target(self, target):
        self.target = float(target)
