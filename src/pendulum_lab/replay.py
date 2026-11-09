"""Feed a recorded hardware episode back through a controller, offline.

The hardware loop builds each observation from the measured state and the
previously sent action, so a recorded episode contains everything needed to
rebuild the exact observations a controller saw. Replaying them answers
"would this code have sent the same commands on that run?" without the rig.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np

from .contracts import PROFILE, observation_frame
from .controllers import load_controller

# The runner fills the history with the pre-engage state, which is not logged.
# Frames older than this tick may differ, so comparisons start here.
WARMUP_TICKS = PROFILE.history


def replay(run_dir, *, repo_root=".", config=None, algorithm=None):
    """Return (replayed_actions, recorded_actions, replayed_modes, recorded_modes).

    `algorithm` and `config` replace the run's own, to ask what another
    controller would have sent on the same observations.
    """
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    args = json.loads((run_dir / "run.json").read_text())["arguments"]
    data = np.load(run_dir / "episode_000.npz", allow_pickle=True)
    samples, timing = data["samples"], data["hardware_timing"]
    checkpoint = args.get("checkpoint")
    controller = load_controller(
        algorithm or args["algorithm"], repo_root / args["params"],
        repo_root / (config or args["config"]),
        repo_root / checkpoint if checkpoint else None)
    controller.reset()
    history = deque([observation_frame(samples[0, 1:5], 0.0)] * PROFILE.history,
                    maxlen=PROFILE.history)
    set_sample_time = getattr(controller, "set_sample_time_us", None)
    actions, modes, previous = [], [], 0.0
    for row, time_us in zip(samples, timing[:, 1]):
        history.append(observation_frame(row[1:5], previous))
        if callable(set_sample_time):
            set_sample_time(int(time_us))
        actions.append(controller.act(np.concatenate(history)))
        modes.append(int(bool(getattr(controller, "balancing", False))))
        previous = row[5]
    return np.asarray(actions), samples[:, 5], np.asarray(modes), timing[:, 6].astype(int)
