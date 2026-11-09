"""Re-zero the pendulum angle at the true hanging rest (motor never engaged).

The firmware's pendulum angle is accumulated from AS5600 changes since the Nano
booted (or was last tared), so its zero is wherever the pendulum hung at that
moment. Hinge friction lets it rest about 1 degree off vertical, and a reboot
while it still swings is worse. A zero error of about 1 degree makes LQR balance
tilted and run the arm into its limit. Let the pendulum hang still (tap the hub
lightly so it settles truly vertical, then hands off), then run this.
"""
import argparse
import math
import time

import numpy as np

from pendulum_lab.serial_client import LowLevelClient


def hanging_offset_deg(client, seconds):
    readings = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        readings.append(client.get_state().pendulum_pos_rad)
        time.sleep(0.02)
    r = np.unwrap(np.asarray(readings))
    return math.degrees(float(np.mean(r))), math.degrees(float(np.std(r)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    with LowLevelClient(args.port, timeout=0.5) as client:
        client.wait_until_ready()
        before, motion = hanging_offset_deg(client, args.seconds)
        print(f"before: hanging reading {before:+.2f} deg (0 = zero is right), movement {motion:.2f} deg rms")
        if motion > 0.3:
            raise SystemExit("The pendulum is moving: let it settle completely and run again.")
        if not client.tare_pendulum():
            raise SystemExit("The board did not acknowledge the tare.")
        time.sleep(0.2)
        after, _ = hanging_offset_deg(client, 1.0)
        print(f"after:  hanging reading {after:+.2f} deg. Zero set at the current rest position.")


if __name__ == "__main__":
    main()
