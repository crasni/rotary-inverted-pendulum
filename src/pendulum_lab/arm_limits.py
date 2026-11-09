"""Optional arm braking envelope shared by hardware and simulation controllers."""
import math

import numpy as np

from .contracts import PROFILE, checked_action, state_from_obs
from .plant.pendulum_env import MOTOR_PLANNER_SPEED_LIMIT_RAD_S


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


class ArmLimitedController:
    """Wrap a controller without its own limiter (SAC, PPO, DQN).

    The classical controllers apply the same limit inside HybridController,
    after approach braking, so load_controller does not wrap them.
    """

    def __init__(self, controller, *, delay_s, speed_rad_s, travel_rad):
        validate_arm_limits(delay_s, speed_rad_s, travel_rad)
        self.controller = controller
        self.delay = delay_s
        self.speed_target = speed_rad_s
        self.travel_target = travel_rad
        self.requested_action = None

    @property
    def balancing(self):
        return getattr(self.controller, "balancing", None)

    @property
    def pre_arm_limit_action(self):
        return self.requested_action

    def reset(self):
        self.controller.reset()
        self.requested_action = None

    def set_sample_time_us(self, sample_time_us):
        set_time = getattr(self.controller, "set_sample_time_us", None)
        if callable(set_time):
            set_time(sample_time_us)

    def act(self, observation):
        arm, _, velocity, _ = state_from_obs(observation)
        self.requested_action = checked_action(self.controller.act(observation))
        return limit_arm_action(self.requested_action, arm, velocity, observation[-1],
                                delay=self.delay, speed_target=self.speed_target,
                                travel_target=self.travel_target)
