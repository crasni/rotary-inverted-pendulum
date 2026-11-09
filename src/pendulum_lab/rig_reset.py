"""Bring the rig back to rest between on-rig training episodes.

The motor must stay engaged for a whole session: the firmware watchdog
latches a fault after 100 ms without a command, and a disengaged stepper can
be backdriven, so its step count would no longer match the physical arm.
This controller therefore runs at 50 Hz between episodes until the start
condition of a normal hardware run holds again: arm centred, pendulum hanging
still.

Pendulum energy E = 0.5*omega**2 + a*cos(theta) obeys dE/dt = b*cos(theta)*omega*u
for arm acceleration u, so u = -c*omega*cos(theta) only ever removes energy
(dE/dt = -c*b*(omega*cos(theta))**2), from a spin down to small swings. A slow
PD recentres the arm, and the shared arm limiter keeps every command inside
the travel and speed envelope. `RigSession.reset` decides when the rig is
close enough to rest and finishes with an exact position move.
"""
import math

import numpy as np

from .arm_limits import limit_arm_action
from .contracts import PROFILE

DAMPING = 4.0           # rad/s^2 of arm acceleration per rad/s of pendulum speed
ARM_KP = 3.0            # rad/s^2 per rad
ARM_KD = 2.5            # rad/s^2 per rad/s
MAX_ACTION = 0.5        # normalized; half the episode authority


class ResetController:
    def __init__(self, *, arm_limits):
        self.arm_limits = arm_limits  # (delay_s, speed_rad_s, travel_rad)

    def reset(self):
        pass

    def act(self, x, previous_action):
        """x = [arm, theta (0 upright), arm speed, pendulum speed]."""
        arm, theta, arm_speed, omega = x
        accel = -DAMPING * omega * math.cos(theta) - ARM_KP * arm - ARM_KD * arm_speed
        action = float(np.clip(accel / PROFILE.max_accel, -MAX_ACTION, MAX_ACTION))
        delay, speed, travel = self.arm_limits
        return limit_arm_action(action, arm, arm_speed, previous_action,
                                delay=delay, speed_target=speed, travel_target=travel)


# -- assisted near-upright starts (training curriculum only) -----------------
LIFT_KICK = ((0.3, 4), (-0.3, 4))   # (action, ticks): start a swing from rest
LIFT_GAIN = 2.5                      # full pump beyond an energy error of 1/LIFT_GAIN
LIFT_PUMP_START = 18.0               # rad/s^2, initial pump strength
LIFT_PUMP_MAX = 45.0                 # rad/s^2, strongest pump the adaptation may reach
LIFT_ADAPT = 15.0                    # rad/s^2 per second per unit energy error
LIFT_PEAK_GAIN = 0.3                 # energy-aim correction per swing, per unit peak shortfall
LIFT_OFFSET_MAX = 0.06
LIFT_ADAPT_ABOVE = 0.3               # only strengthen while this far below the target
LIFT_LEAD_S = 0.04
LIFT_PHASE_WIDTH = 1.0               # rad/s of omega*cos(theta) over which the pump reverses
LIFT_ARM_KP = 6.0
LIFT_ARM_KD = 4.0
LIFT_COAST_KP = 6.0
LIFT_COAST_KD = 4.0
LIFT_MAX_ACTION = 0.95


class LiftController:
    """Swing the pendulum from rest to a chosen energy, arm kept near centre.

    Used only to give a learning policy near-upright starting states; it is a
    reset tool like `ResetController`, never a teacher (its ticks are not
    training data) and never part of an evaluation. With normalized energy
    e = 0.5*(omega/w0)**2 + cos(theta) (1 = upright at rest, -1 = hanging),
    a target e* = cos(peak) makes the pendulum rise to `peak` from upright.
    Inside `tolerance` of e* it stops pumping and only recentres the arm.
    """

    def __init__(self, *, arm_limits, target_energy, tolerance=0.06, natural_rad_s=12.27):
        self.arm_limits = arm_limits
        self.target_energy = target_energy
        self.tolerance = tolerance
        self.w0 = natural_rad_s
        self.tick = 0
        self.strength = LIFT_PUMP_START
        self.offset = 0.0
        self.last_omega = 0.0

    def reset(self):
        self.tick = 0
        self.strength = LIFT_PUMP_START
        self.offset = 0.0
        self.last_omega = 0.0

    def energy(self, x):
        return 0.5 * (x[3] / self.w0) ** 2 + math.cos(x[1])

    def act(self, x, previous_action):
        arm, theta, arm_speed, omega = x
        k, self.tick = self.tick, self.tick + 1
        for action, ticks in LIFT_KICK:
            if k < ticks:
                return self._limit(action, x, previous_action)
            k -= ticks
        # Per swing: where the pendulum actually turns around near the top
        # tells how far the energy estimate is from the reached height; aim
        # higher (or lower) by the difference.
        if omega * self.last_omega < 0 and math.cos(theta) > 0:
            self.offset = float(np.clip(self.offset + LIFT_PEAK_GAIN * (self.target_energy - math.cos(theta)),
                                        -LIFT_OFFSET_MAX, LIFT_OFFSET_MAX))
        self.last_omega = omega
        error = self.target_energy + self.offset - self.energy(x)
        # Rig losses differ from the simulator's: strengthen the pump while the
        # energy stays short of the target, weaken it after an overshoot.
        if error > LIFT_ADAPT_ABOVE or error < 0:
            self.strength = float(np.clip(self.strength + LIFT_ADAPT * error * PROFILE.dt,
                                          LIFT_PUMP_START, LIFT_PUMP_MAX))
        if abs(error) > self.tolerance:
            # Pump on the swing predicted LIFT_LEAD_S ahead, to cover the command delay.
            lead_theta = theta + omega * LIFT_LEAD_S
            lead_omega = omega + self.w0 ** 2 * math.sin(theta) * LIFT_LEAD_S
            phase = math.tanh(lead_omega * math.cos(lead_theta) / LIFT_PHASE_WIDTH)
            pump = self.strength * float(np.clip(LIFT_GAIN * error, -1.0, 1.0)) * phase
            accel = pump - LIFT_ARM_KP * arm - LIFT_ARM_KD * arm_speed
        else:
            accel = -LIFT_COAST_KP * arm - LIFT_COAST_KD * arm_speed
        return self._limit(float(np.clip(accel / PROFILE.max_accel, -LIFT_MAX_ACTION, LIFT_MAX_ACTION)),
                           x, previous_action)

    def _limit(self, action, x, previous_action):
        delay, speed, travel = self.arm_limits
        return limit_arm_action(action, x[0], x[2], previous_action,
                                delay=delay, speed_target=speed, travel_target=travel)
