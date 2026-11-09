"""Cascaded PID balance with energy or SAC swing-up; see docs/algorithms.md.

The balance law is the `Controller` class at the end of this file. Everything
above it is the shared hand-off block. Without delay settings it balances the
4-state model and can use balance_state_lead_s. With `delay` (and optionally
`pendulum_input_delay_s`) it first predicts the state until the commands still
in flight have acted, then applies the PID loop to that predicted state.
"""
# ===== BEGIN SHARED HAND-OFF =====
# Swing-up, capture, state estimation and arm limits shared by lqr.py,
# mpc.py, ilqr.py and pid.py. The assignment keeps each algorithm in its own
# file, so each file carries this identical copy. Edit it in lqr.py, then run
#     .venv/bin/python scripts/sync_handoff.py
# tests/test_handoff_sync.py fails if the copies differ.
# Settings reference: docs/controller-settings.md
from collections import deque
import math

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.signal import cont2discrete

from pendulum_lab.contracts import PROFILE, state_from_obs, wrap
from pendulum_lab.plant.pendulum_env import (
    ARM_LENGTH_M, GRAVITY, MOTOR_PLANNER_SPEED_LIMIT_RAD_S, PendulumParams,
)


# --- Linearized model ---------------------------------------------------
def plant(params_path):
    """Return (params, a, b, damping, A, B) for x = [arm, theta, arm_vel, theta_vel].

    The input is the normalized arm acceleration command in [-1, 1].
    Near upright: theta_ddot = a*theta + b*arm_accel - damping*theta_dot.
    """
    p = PendulumParams.load(params_path)
    a = p.pendulum_mass_kg * GRAVITY * p.pendulum_com_m / p.pendulum_inertia_kg_m2
    b = p.pendulum_mass_kg * ARM_LENGTH_M * p.pendulum_com_m / p.pendulum_inertia_kg_m2
    damping = p.pendulum_friction / p.pendulum_inertia_kg_m2
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, a, 0, -damping]])
    B = np.array([[0], [0], [1], [b]]) * PROFILE.max_accel
    return p, a, b, damping, A, B


def discretize(A, B, dt=PROFILE.dt):
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), dt)
    return Ad, Bd


def delayed_model(A, B, delay, dt=PROFILE.dt):
    """Exact ZOH discretization with an input delay, augmented with past actions.

    State z = [x, u_{k-1}, ..., u_{k-n}], where n = ceil(delay / dt).
    """
    def flow(T):
        M = np.zeros((5, 5))
        M[:4, :4], M[:4, 4:] = A, B
        E = expm(M * T)
        return E[:4, :4], E[:4, 4:]
    n = int(np.ceil(delay / dt - 1e-9))
    frac = delay - (n - 1) * dt
    Ad, _ = flow(dt)
    E1, G1 = flow(dt - frac)
    _, Gf = flow(frac)
    newer, older = G1, E1 @ Gf
    Az, Bz = np.zeros((4 + n, 4 + n)), np.zeros((4 + n, 1))
    Az[:4, :4] = Ad
    if n == 1:
        Bz[:4] = newer
    else:
        Az[:4, 4 + n - 2:4 + n - 1] = newer
    Az[:4, 4 + n - 1:] = older
    Bz[4, 0] = 1.0
    for i in range(1, n):
        Az[4 + i, 4 + i - 1] = 1.0
    return Az, Bz, n


def balance_model(A, B, arm_delay, pendulum_delay):
    """Delay-augmented model where the arm and pendulum respond after different delays."""
    if arm_delay == pendulum_delay:
        return delayed_model(A, B, arm_delay)

    # Both channels share the same command history, but affect different
    # equations at different times. Superpose only the physical-state blocks.
    arm_input = B.copy()
    arm_input[3, 0] = 0.0
    arm_A, arm_B, arm_history = delayed_model(A, arm_input, arm_delay)
    combined_A, combined_B, history = delayed_model(A, B - arm_input, pendulum_delay)
    combined_A[:4, 4:4 + arm_history] += arm_A[:4, 4:]
    combined_B[:4] += arm_B[:4]
    return combined_A, combined_B, history


# --- Arm command limiter (same as pendulum_lab.arm_limits) ----------------
def limit_arm_action(action, arm, velocity, previous_action, *, delay, speed_target, travel_target):
    arrival_velocity = velocity + previous_action * PROFILE.max_accel * delay
    # v*reaction + v**2/(2*braking) must fit in the remaining travel.
    room = np.maximum(0.0, travel_target - np.array([arm, -arm]))
    braking = 0.5 * PROFILE.max_accel
    reaction = delay + PROFILE.dt
    speed = np.sqrt((braking * reaction)**2 + 2 * braking * room) - braking * reaction
    forward, reverse = np.minimum(speed, speed_target)
    scale = 2 * PROFILE.dt * PROFILE.max_accel
    lower, upper = np.clip([(-reverse - arrival_velocity) / scale,
                            (forward - arrival_velocity) / scale], -1.0, 1.0)
    return float(np.clip(action, lower, upper))


def validate_arm_limits(delay_s, speed_rad_s, travel_rad):
    if not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError("Arm command delay must be finite and nonnegative")
    if not 0 < speed_rad_s < MOTOR_PLANNER_SPEED_LIMIT_RAD_S:
        raise ValueError("Arm command speed must be below the firmware planner limit")
    if not 0 < travel_rad < math.radians(PROFILE.arm_limit_deg):
        raise ValueError("Arm command travel must be inside the hardware guard")


# --- Energy swing-up -------------------------------------------------------
# A pendulum hanging still has no energy for the energy law to amplify, so it
# gets alternating kicks of this size (rad/s^2) every KICK_PERIOD_TICKS.
KICK_ACCEL = 12.0
KICK_PERIOD_TICKS = 25
KICK_MAX_SPEED_RAD_S = 0.15
KICK_HANGING_COS = -0.95
# Brake the arm toward center beyond this angle during swing-up.
SWING_ARM_BRAKE_RAD = 1.15

# --- Adaptive energy target (`swing_energy_max_ratio`) ----------------------
# One "excursion" starts when |theta| drops below ENTER and ends above EXIT.
ADAPT_ENTER_RAD, ADAPT_EXIT_RAD = 1.2, 1.4
ADAPT_SHORT_RAD = 0.23          # closest approach further than this: add energy
ADAPT_FAST_RAD = 0.18           # closer than this and faster than FAST_SPEED: remove energy
ADAPT_FAST_SPEED_RAD_S = 5.5
ADAPT_GOOD_SPEED_RAD_S = 4.0    # crossing speed the reduction aims for
ADAPT_MAX_STEP = 0.10
ADAPT_MIN_RATIO = 0.98

# --- Approach braking (learned swing-up only) -------------------------------
BRAKE_ACTIVE_RAD = 0.9          # only brake inside this angle on an inward approach
BRAKE_MIN_LEVERAGE = 0.08       # floor on |sin(theta)| to keep the bound finite

# --- Velocity observer (`balance_observer`) ---------------------------------
OBSERVER_WARMUP_SAMPLES = 7     # consecutive regular samples before estimating
OBSERVER_FIT_SAMPLES = 5        # samples used by the least-squares fit
OBSERVER_MAX_ANGLE_RAD = 0.7    # stop estimating beyond this angle
OBSERVER_WEIGHT = 0.4           # share of the correction applied at low residual
OBSERVER_RESIDUAL_RAD = 0.018   # residual above which the weight shrinks
OBSERVER_MAX_CORRECTION_RAD_S = 0.7
OBSERVER_USE_ANGLE_RAD = 0.30   # only replace the measured speed inside this angle


# Settings that describe the rig and the hand-off rather than one balance law.
# The transfer presets copy these from the LQR hardware reference unchanged.
HANDOFF_KEYS = (
    "swing_up", "approach_brake_speed_rad_s", "arm_recenter_rate_rad_s",
    "balance_observer", "balance_velocity_blend", "capture_angle", "capture_speed",
    "capture_arm_angle", "capture_arm_speed", "capture_max_action",
)


def swing_action(x, tick, a, b, config):
    """Energy pumping toward E/I = a * swing_target_energy_ratio, with arm centering."""
    # E/I = .5*theta_dot**2 + a*cos(theta); upright at rest has E/I = a.
    arm, theta, velocity, omega = x
    target = a * config.get("swing_target_energy_ratio", 1.0)
    error = target - (0.5 * omega**2 + a * np.cos(theta))
    drive = config.get("energy_gain", 0.1) * error * omega * np.cos(theta) / max(b, 1e-6)
    if (drive_cap := config.get("swing_drive_cap")) is not None:
        # Reserve acceleration for arm centering before clipping the final command.
        drive = np.clip(drive, -drive_cap, drive_cap)
    if abs(omega) < KICK_MAX_SPEED_RAD_S and np.cos(theta) < KICK_HANGING_COS:
        drive += KICK_ACCEL * (1 if (tick // KICK_PERIOD_TICKS) % 2 == 0 else -1)
    accel = drive - config.get("swing_arm_kp", 3.0) * arm - config.get("swing_arm_kd", 2.0) * velocity
    # Brake before the travel boundary; this is a controller heuristic, not a hardware guard.
    if abs(arm) > SWING_ARM_BRAKE_RAD and arm * velocity > 0:
        accel = -np.sign(arm) * min(50.0, 12.0 + 12.0 * abs(velocity))
    return float(np.clip(accel / PROFILE.max_accel, -1, 1))


def brake_approach(action, x, previous_action, a, b, damping, approach_speed):
    """Limit a swing-up command so the pendulum crosses upright near approach_speed."""
    _, theta, velocity, omega = x
    if abs(theta) >= BRAKE_ACTIVE_RAD or theta * omega >= 0.0:
        return action

    # Predict the queued command's effect before the new action takes effect.
    speed_limit = MOTOR_PLANNER_SPEED_LIMIT_RAD_S
    pending = np.clip(previous_action * PROFILE.max_accel,
                      (-speed_limit - velocity) / PROFILE.dt,
                      (speed_limit - velocity) / PROFILE.dt)
    sin, cos = np.sin(theta), np.cos(theta)
    alpha = a * sin + b * cos * pending + velocity**2 * sin * cos - damping * omega
    future_theta = theta + PROFILE.dt * (omega + 0.5 * PROFILE.dt * alpha)
    future_omega = omega + PROFILE.dt * alpha
    excess = 0.5 * (future_omega**2 - approach_speed**2) + a * (np.cos(future_theta) - 1.0)
    if excess <= 0.0:
        return action

    # Energy added per unit inertia is b*u*delta(sin(theta)).
    leverage = max(BRAKE_MIN_LEVERAGE, abs(np.sin(future_theta)))
    maximum_inward_effort = -excess / (b * PROFILE.max_accel * leverage)
    direction = np.sign(omega)
    return float(direction * np.clip(min(direction * action, maximum_inward_effort), -1.0, 1.0))


def frame_angle(observation, age):
    """Pendulum angle from the frame `age` ticks before the newest one."""
    start = -6 * (age + 1)
    return np.arctan2(observation[start + 1], observation[start + 2])


class HybridController:
    """Base class. Subclasses set `section` and implement the balance law."""

    section = None                # config key holding this algorithm's settings
    default_delay_s = 0.025       # arm command delay when "delay" is absent
    always_delayed_model = False  # build the delay-augmented model even without delay keys
    supports_input_delay = True   # False: the balance law only accepts the 4-state
    handles_arm_command_limits = True  # tells load_controller not to wrap again

    def __init__(self, *, params_path, config, checkpoint=None):
        self.config = cfg = config.get(self.section, config)
        self.p, self.a, self.b, self.damping, Ac, Bc = plant(params_path)

        # Model and delays.
        if "arm_input_delay_s" in cfg:
            raise ValueError("arm_input_delay_s was renamed to delay")
        has_delay = "delay" in cfg or "pendulum_input_delay_s" in cfg
        self.delay = cfg.get("delay", self.default_delay_s)
        pendulum_delay = cfg.get("pendulum_input_delay_s", self.delay)
        if "pendulum_input_delay_s" in cfg and not (
                0.0 < self.delay <= pendulum_delay <= PROFILE.history * PROFILE.dt):
            raise ValueError("pendulum_input_delay_s must be at least delay and fit the observation history")
        coupling_scale = cfg.get("balance_coupling_scale", 1.0)
        if not math.isfinite(coupling_scale) or coupling_scale <= 0.0:
            raise ValueError("balance_coupling_scale must be finite and positive")
        lead = cfg.get("balance_state_lead_s", 0.0)
        if not 0.0 <= lead <= PROFILE.dt:
            raise ValueError("balance_state_lead_s must be between zero and one control tick")
        delayed = self.always_delayed_model or (has_delay and self.supports_input_delay)
        if not self.supports_input_delay and (
                "pendulum_input_delay_s" in cfg or coupling_scale != 1.0):
            raise ValueError(f"{self.section} balances the 4-state model; use balance_state_lead_s "
                             "instead of pendulum_input_delay_s or balance_coupling_scale")
        if delayed and lead:
            raise ValueError("The delay-augmented model already accounts for queued commands; "
                             "remove balance_state_lead_s")
        Bc[3, 0] *= coupling_scale
        if delayed:
            self.A, self.B, self.n_past = balance_model(Ac, Bc, self.delay, pendulum_delay)
        else:
            self.A, self.B = discretize(Ac, Bc)
            self.n_past = 0
        self.lead = discretize(Ac, Bc, lead) if lead else None

        # Hand-off and state estimation.
        self.velocity_blend = cfg.get("balance_velocity_blend", 0.0)
        if not 0.0 <= self.velocity_blend <= 1.0:
            raise ValueError("balance_velocity_blend must be in [0, 1]")
        self.use_balance_observer = cfg.get("balance_observer", False)
        self.recenter_rate = cfg.get("arm_recenter_rate_rad_s", 0.0)
        if not math.isfinite(self.recenter_rate) or self.recenter_rate < 0.0:
            raise ValueError("arm_recenter_rate_rad_s must be finite and nonnegative")
        self.recenter_max = cfg.get("arm_recenter_max_rad", 0.64)
        # Recentering and the observer pause above this arm speed so the
        # balance law keeps the speed headroom below the arm limiter.
        self.arm_speed_headroom = cfg.get("balance_arm_speed_headroom_rad_s", 3.6)

        # Swing-up.
        self.swing_energy_max = cfg.get("swing_energy_max_ratio")
        if self.swing_energy_max is not None:
            target = cfg.get("swing_target_energy_ratio", 1.0)
            if not np.isfinite([target, self.swing_energy_max]).all() or not 0.98 <= target <= self.swing_energy_max:
                raise ValueError("Adaptive swing energy requires 0.98 <= initial target <= finite maximum")
        self.swing_energy_mix = cfg.get("swing_energy_mix", 0.0)
        if not 0.0 <= self.swing_energy_mix <= 1.0:
            raise ValueError("swing_energy_mix must be in [0, 1]")
        self.approach_speed = cfg.get("approach_brake_speed_rad_s")
        self.learned_swing = None
        if cfg.get("swing_up") == "sac":
            if checkpoint is None:
                raise ValueError(f"{self.section} with SAC swing-up requires --checkpoint")
            from pendulum_lab.algorithms.sac import Controller as SAC
            self.learned_swing = SAC(params_path=params_path, config=config, checkpoint=checkpoint)
        if self.approach_speed is not None:
            if self.learned_swing is None:
                raise ValueError("approach_brake_speed_rad_s only applies to SAC swing-up")
            if not math.isfinite(self.approach_speed) or self.approach_speed <= 0:
                raise ValueError("Approach braking speed must be finite and positive")

        self.arm_limits = self.read_arm_limits(config, cfg)

        self.setup_balance()
        self.reset()

    # -- configuration ---------------------------------------------------
    def read_arm_limits(self, config, cfg):
        """Return (delay_s, speed_rad_s, travel_rad) or None.

        Preferred: top-level "arm_command_limits": {"delay_s", "speed_rad_s", "travel_rad"}.
        Older presets set top-level "arm_speed_target_rad_s" and
        "arm_travel_target_rad" and reuse the section's "delay".
        SAC swing-up always runs with limits (4.7 rad/s, 1.42 rad by default).
        """
        block = config.get("arm_command_limits")
        old = [k for k in ("arm_speed_target_rad_s", "arm_travel_target_rad") if k in config]
        if block is not None and old:
            raise ValueError("Use arm_command_limits or the older arm_*_target keys, not both")
        if block is not None:
            limits = block["delay_s"], block["speed_rad_s"], block["travel_rad"]
        elif old or self.learned_swing is not None:
            limits = (self.delay, config.get("arm_speed_target_rad_s", 4.7),
                      config.get("arm_travel_target_rad", 1.42))
        else:
            return None
        validate_arm_limits(*limits)
        return limits

    # -- subclass interface ------------------------------------------------
    def setup_balance(self):
        """Build gains from self.A, self.B (4 + n_past states). Must set self.K."""
        raise NotImplementedError

    def balance_command(self, x, observation, *, use_observer):
        """Return the normalized balance command (unclipped is fine)."""
        return self.balance(self.balance_state(x, observation, use_observer=use_observer))

    def capture_command(self, x, observation):
        """Command the capture gate checks against capture_max_action."""
        return (-self.K @ self.balance_state(x, observation)).item()

    def on_release(self):
        self.integral = 0.0

    # -- runtime -----------------------------------------------------------
    def reset(self):
        self.tick = 0
        self.balancing = False
        self.arm_reference_rad = 0.0
        self.balance_history = (deque(maxlen=OBSERVER_WARMUP_SAMPLES)
                                if self.use_balance_observer else None)
        self.observer_time = 0.0
        self.sample_time_us = self.previous_sample_time_us = None
        self.estimated_pendulum_velocity_rad_s = None
        self.integral = 0.0
        self.plan = np.zeros(int(self.config.get("horizon", 25)))
        self.swing_config = self.config if self.swing_energy_max is None else self.config.copy()
        self.swing_peak = None
        if self.learned_swing is not None:
            self.learned_swing.reset()
        if self.arm_limits is not None:
            self.requested_action = 0.0
            self.pre_arm_limit_action = 0.0

    def act(self, observation):
        cfg = self.config
        x = state_from_obs(observation)
        self.tick += 1
        was_balancing = self.balancing
        if self.use_balance_observer:
            self.update_balance_observer(x)
        # Choose the arm reference before checking the first balance command.
        if (abs(x[1]) < cfg.get("capture_angle", 0.5)
                and abs(x[3]) < cfg.get("capture_speed", 2.0)
                and abs(x[0]) < cfg.get("capture_arm_angle", np.inf)
                and abs(x[2]) < cfg.get("capture_arm_speed", np.inf)):
            if not self.balancing:
                if self.recenter_rate:
                    self.arm_reference_rad = float(np.clip(x[0], -self.recenter_max, self.recenter_max))
                limit = cfg.get("capture_max_action", np.inf)
                self.balancing = (limit == np.inf
                                  or abs(self.capture_command(x, observation)) <= limit)
        elif abs(x[1]) > cfg.get("release_angle", 0.55):
            self.balancing = False
            self.on_release()
        if not self.balancing:
            self.arm_reference_rad = 0.0
            if self.learned_swing is None:
                if self.swing_energy_max is not None:
                    self.update_swing_target(abs(x[1]), abs(x[3]))
                action = swing_action(x, self.tick, self.a, self.b, self.swing_config)
            else:
                action = self.learned_swing.act(observation)
                if self.swing_energy_mix:
                    energy_action = swing_action(x, self.tick, self.a, self.b, self.swing_config)
                    action += self.swing_energy_mix * (energy_action - action)
        else:
            self.swing_peak = None
            if self.recenter_rate and abs(x[2]) < self.arm_speed_headroom:
                step = self.recenter_rate * PROFILE.dt
                self.arm_reference_rad -= float(np.clip(self.arm_reference_rad, -step, step))
            action = float(np.clip(self.balance_command(x, observation, use_observer=was_balancing), -1, 1))
        if self.arm_limits is None:
            return action
        self.requested_action = action
        if not self.balancing and self.approach_speed is not None:
            action = brake_approach(action, x, observation[-1], self.a, self.b,
                                    self.damping, self.approach_speed)
        self.pre_arm_limit_action = action
        delay, speed, travel = self.arm_limits
        return limit_arm_action(action, x[0], x[2], observation[-1],
                                delay=delay, speed_target=speed, travel_target=travel)

    def balance_state(self, x, observation, *, use_observer=False):
        """State handed to the balance law: [x, u_{k-1}, ..., u_{k-n_past}].

        Applies the arm reference, the velocity observer or blend, and the
        optional one-tick state prediction.
        """
        # Frames run oldest to newest; each ends with the action sent before it.
        # observation[-1] is u[k-1], observation[-7] is u[k-2], and so on.
        past = [observation[-1 - 6 * i] for i in range(self.n_past)]
        state = np.r_[x, past]
        state[0] -= self.arm_reference_rad
        # Keep the capture check and initial command on the established
        # feedback path. Use the observer once upright and away from the rail.
        if (use_observer and self.estimated_pendulum_velocity_rad_s is not None
                and abs(x[1]) < OBSERVER_USE_ANGLE_RAD and abs(x[2]) < self.arm_speed_headroom):
            state[3] = self.estimated_pendulum_velocity_rad_s
        elif self.velocity_blend:
            theta, omega = x[1], x[3]
            previous, older = frame_angle(observation, 1), frame_angle(observation, 2)
            # The three-point backward derivative estimates velocity at the
            # latest angle sample. Wrap each increment across the angle seam.
            estimate = (3 * wrap(theta - previous) - wrap(previous - older)) / (2 * PROFILE.dt)
            state[3] += self.velocity_blend * (estimate - omega)
        if self.lead is not None:
            transition, command = self.lead
            state = transition @ state + command[:, 0] * observation[-1]
        return state

    def set_sample_time_us(self, sample_time_us):
        if self.use_balance_observer:
            self.sample_time_us = sample_time_us

    def update_balance_observer(self, x):
        interval = PROFILE.dt
        if self.sample_time_us is not None and self.previous_sample_time_us is not None:
            elapsed = ((self.sample_time_us - self.previous_sample_time_us) % (2**32)) * 1e-6
            if 0.5 * PROFILE.dt <= elapsed <= 2.0 * PROFILE.dt:
                interval = elapsed
            else:
                self.balance_history.clear()
        self.previous_sample_time_us = self.sample_time_us
        self.sample_time_us = None
        self.observer_time += interval
        self.balance_history.append((self.observer_time, x[1], x[2], x[3]))
        self.estimated_pendulum_velocity_rad_s = None
        if (len(self.balance_history) < self.balance_history.maxlen
                or abs(x[1]) >= OBSERVER_MAX_ANGLE_RAD):
            return

        time, theta, velocity, omega = np.asarray(self.balance_history)[-OBSERVER_FIT_SAMPLES:].T
        theta = np.unwrap(theta)
        midpoint = (time[1:] + time[:-1]) / 2
        angle = (theta[1:] + theta[:-1]) / 2
        arm_speed = (velocity[1:] + velocity[:-1]) / 2
        drift = (self.a * np.sin(angle) + arm_speed**2 * np.sin(angle) * np.cos(angle)
                 - self.damping * (omega[1:] + omega[:-1]) / 2)
        impulse = drift * np.diff(time) + self.b * np.cos(angle) * np.diff(velocity)

        # Integrate the angle equation backwards from the newest sample.
        # Each past angle gives one estimate of the current angular velocity.
        span = time[-1] - time[:-1]
        moments = np.maximum(midpoint[None, :] - time[:-1, None], 0.0) @ impulse
        displacement = theta[-1] - theta[:-1] + moments
        estimate = (span @ displacement) / (span @ span)
        residual = np.sqrt(np.mean((displacement - span * estimate)**2))
        weight = OBSERVER_WEIGHT * min(1.0, OBSERVER_RESIDUAL_RAD / max(residual, OBSERVER_RESIDUAL_RAD))
        correction = np.clip(weight * (estimate - x[3]),
                             -OBSERVER_MAX_CORRECTION_RAD_S, OBSERVER_MAX_CORRECTION_RAD_S)
        self.estimated_pendulum_velocity_rad_s = float(x[3] + correction)

    def update_swing_target(self, angle, speed):
        # Track one excursion near upright. Separate entry and exit angles keep
        # quantized feedback from updating the target repeatedly at a boundary.
        if angle < ADAPT_ENTER_RAD and (self.swing_peak is None or angle < self.swing_peak[0]):
            self.swing_peak = (angle, speed)
        if angle <= ADAPT_EXIT_RAD or self.swing_peak is None:
            return
        nearest, speed_at_peak = self.swing_peak
        self.swing_peak = None
        target = self.swing_config.get("swing_target_energy_ratio", 1.0)
        if nearest > ADAPT_SHORT_RAD:
            # Potential-energy shortfall divided by m*g*l.
            target = min(self.swing_energy_max, target + min(ADAPT_MAX_STEP, 1.0 - np.cos(nearest)))
        elif nearest < ADAPT_FAST_RAD and speed_at_peak > ADAPT_FAST_SPEED_RAD_S:
            # Remove excess kinetic energy above the target approach speed.
            excess = (speed_at_peak**2 - ADAPT_GOOD_SPEED_RAD_S**2) / (2.0 * self.a)
            target = max(ADAPT_MIN_RATIO, target - min(ADAPT_MAX_STEP, excess))
        self.swing_config["swing_target_energy_ratio"] = target
# ===== END SHARED HAND-OFF =====


class Controller(HybridController):
    section = "pid"

    def setup_balance(self):
        # The LQR gain serves the capture check.
        self.Q = np.zeros_like(self.A)
        self.Q[:4, :4] = np.diag(self.config.get("q", [2.0, 60.0, 1.0, 2.0]))
        self.R = np.array([[self.config.get("r", 0.5)]])
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R + self.B.T @ self.P @ self.B, self.B.T @ self.P @ self.A)
        # With `delay` set, the state carries the commands still in flight.
        # Predicting until they have all acted (a Smith-style predictor) gives
        # the PID loop the state its next command will actually meet.
        self.predictor = (np.linalg.matrix_power(self.A, self.n_past)[:4]
                          if self.n_past else None)

    def balance(self, x):
        if self.predictor is not None:
            x = self.predictor @ x
        arm, theta, velocity, omega = x
        # Outer arm loop requests a small pendulum lean. The sign follows
        # theta_ddot = a*theta + b*arm_accel near upright.
        target = np.clip(self.config.get("arm_kp", 0.03) * arm
                         + self.config.get("arm_kd", 0.03) * velocity, -0.15, 0.15)
        error = theta - target
        candidate = np.clip(self.integral + error * PROFILE.dt, -0.3, 0.3)
        accel = -(self.config.get("kp", 1000.0) * error
                  + self.config.get("kd", 90.0) * omega
                  + self.config.get("ki", 2.0) * candidate)
        if abs(accel) < PROFILE.max_accel:
            self.integral = candidate
        return accel / PROFILE.max_accel
