"""Watch the pendulum encoder for faults without ever engaging the motor.

Reads the firmware state at 50 Hz and reports the moment it turns non-finite,
with the latched fault bits. Phase 1 runs with 12 V off; then you switch 12 V
on (motor stays disengaged) and it keeps watching. This tells apart an encoder
or wiring problem that is always there from interference that starts when the
motor driver is powered.
"""
import argparse
import math
import time

from pendulum_lab.serial_client import LowLevelClient

FAULTS = ((1, "travel"), (2, "encoder"), (4, "watchdog"), (8, "invalid command"))


def watch(client, seconds, label):
    start = time.monotonic()
    last_print = -1
    previous = None
    while time.monotonic() - start < seconds:
        s = client.get_state()
        t = time.monotonic() - start
        if not all(math.isfinite(v) for v in (s.motor_pos_rad, s.pendulum_pos_rad, s.pendulum_vel_rad_s)):
            bits = client.get_safety_faults()
            names = ", ".join(n for b, n in FAULTS if bits & b) or "none latched"
            print(f"[{label}] FAULT after {t:.1f} s: fault bits={bits} ({names})", flush=True)
            return False
        if int(t) != last_print:
            last_print = int(t)
            jump = "" if previous is None else f", change since last read {math.degrees(s.pendulum_pos_rad - previous):+.2f} deg"
            print(f"[{label}] {t:4.0f} s  pendulum {math.degrees(s.pendulum_pos_rad):+7.2f} deg{jump}", flush=True)
        previous = s.pendulum_pos_rad
        time.sleep(0.02)
    print(f"[{label}] no fault in {seconds:.0f} s", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()
    with LowLevelClient(args.port, timeout=0.1) as client:
        if not client.wait_until_ready():
            raise SystemExit("Firmware READY handshake failed")
        bits = client.get_safety_faults()
        print(f"fault bits at connect: {bits}", flush=True)
        if bits:
            raise SystemExit("A fault is already latched: unplug USB for 5 s and try again")
        print("Phase 1: keep 12 V OFF. Gently swing the pendulum by hand now and then.", flush=True)
        if not watch(client, args.seconds, "12V off"):
            return
        input("Phase 2: switch 12 V ON (the motor stays disengaged), then press Enter: ")
        watch(client, args.seconds, "12V on")
        print("Done. Switch 12 V OFF.")


if __name__ == "__main__":
    main()
