"""Optional experiments built on the learned-swing-up LQR controller.

Default: SAC swing-up with LQR capture and balance.

With "balance_policy" (a SAC checkpoint) in the "lqr" section, the same SAC
swing-up, approach braking, capture gate, release and arm limits are kept and
only the balance law is replaced by the SAC policy. This is a diagnostic: it
separates whether a learned policy can balance on the rig from whether it can
arrive and capture. "balance_observation" chooses what the policy sees:

  raw       the observation the standalone policy sees
  observer  the newest pendulum speed replaced by LQR's observer/blend estimate
  matched   observer, plus arm angles relative to LQR's recentering reference

With "balance_dither": {"amplitude", "seed"} the LQR balance command gets a
known random-telegraph perturbation, for closed-loop identification of the
near-upright dynamics (runs/obs_model/dither_iv.py uses it as an instrument).
"""

import numpy as np

from pendulum_lab.algorithms.lqr import Controller as LQR

MODES = ("raw", "observer", "matched")
ARM, PENDULUM_SPEED, FRAME = 0, 4, 6


class SACBalanceLQR(LQR):
    def __init__(self, *, params_path, config, checkpoint, balance_policy, mode):
        if mode not in MODES:
            raise ValueError(f"balance_observation must be one of {MODES}")
        super().__init__(params_path=params_path, config=config, checkpoint=checkpoint)
        from pendulum_lab.algorithms.sac import Controller as SAC
        self.balance_policy = SAC(params_path=params_path, config=config, checkpoint=balance_policy)
        self.mode = mode

    def balance_command(self, x, observation, *, use_observer):
        observation = np.array(observation, dtype=np.float32)
        if self.mode != "raw":
            # Same estimate the LQR balance law would use this tick.
            state = self.balance_state(x, observation, use_observer=use_observer)
            observation[-FRAME + PENDULUM_SPEED] = state[3]
        if self.mode == "matched":
            observation[ARM::FRAME] -= self.arm_reference_rad
        return self.balance_policy.act(observation)


def dither_sequence(seed, amplitude, length=2000):
    """Random-telegraph balance perturbation, +-amplitude held for 1-3 ticks.

    Indexed by the number of balancing ticks so far, so analysis can rebuild
    it from the logged balance mode without extra log columns.
    """
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < length:
        values += [float(rng.choice((-amplitude, amplitude)))] * int(rng.integers(1, 4))
    return np.array(values[:length])


class DitherLQR(LQR):
    """LQR balance plus a known perturbation, for closed-loop identification
    of the near-upright dynamics (the dither is an instrumental variable)."""

    def __init__(self, *, params_path, config, checkpoint, dither):
        super().__init__(params_path=params_path, config=config, checkpoint=checkpoint)
        amplitude = float(dither["amplitude"])
        if not 0.0 < amplitude <= 0.3:
            raise ValueError("balance dither amplitude must be in (0, 0.3]")
        self.dither = dither_sequence(int(dither.get("seed", 0)), amplitude)

    def reset(self):
        super().reset()
        self.balance_ticks = 0

    def balance_command(self, x, observation, *, use_observer):
        command = super().balance_command(x, observation, use_observer=use_observer)
        perturbation = self.dither[min(self.balance_ticks, len(self.dither) - 1)]
        self.balance_ticks += 1
        return command + perturbation


class Controller:
    def __init__(self, *, params_path, config, checkpoint=None):
        self.lqr = None
        if checkpoint is not None:
            lqr_config = {**config.get("lqr", config), "swing_up": "sac"}
            config = {**config, "lqr": lqr_config}
            balance_policy = lqr_config.get("balance_policy")
            if "balance_dither" in lqr_config:
                if balance_policy is not None:
                    raise ValueError("balance_dither applies to the LQR balance law only")
                self.lqr = DitherLQR(params_path=params_path, config=config, checkpoint=checkpoint,
                                     dither=lqr_config["balance_dither"])
            elif balance_policy is None:
                self.lqr = LQR(params_path=params_path, config=config, checkpoint=checkpoint)
            else:
                self.lqr = SACBalanceLQR(params_path=params_path, config=config, checkpoint=checkpoint,
                                         balance_policy=balance_policy,
                                         mode=lqr_config.get("balance_observation", "raw"))

    def reset(self):
        if self.lqr is not None:
            self.lqr.reset()

    @property
    def balancing(self):
        return self.lqr is not None and self.lqr.balancing

    def __getattr__(self, name):
        # Logging and timing hooks (requested_action, pre_arm_limit_action,
        # arm_reference_rad, estimated_pendulum_velocity_rad_s,
        # set_sample_time_us) come from the wrapped controller.
        if name in ("requested_action", "pre_arm_limit_action", "arm_reference_rad",
                    "estimated_pendulum_velocity_rad_s", "set_sample_time_us"):
            lqr = self.__dict__.get("lqr")
            if lqr is not None and hasattr(lqr, name):
                return getattr(lqr, name)
            return None
        raise AttributeError(name)

    def act(self, observation):
        if self.lqr is None:
            raise NotImplementedError("Pass a trained SAC checkpoint to use algorithms/custom.py")
        return self.lqr.act(observation)


def train(*, env_factory, config, output_dir):
    raise NotImplementedError("Implement CUSTOM training if your method needs it")
