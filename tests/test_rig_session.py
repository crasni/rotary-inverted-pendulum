import math
import unittest
from pathlib import Path

from pendulum_lab.hardware import validate_state
from pendulum_lab.contracts import wrap
from pendulum_lab.rig_session import AssistedStart, RigSession, SessionStop, SimClient, swing_amplitude

PARAMS = Path(__file__).resolve().parents[1] / "configs" / "my_rig.json"
LIMITS = (0.018, 4.4, 1.35)


def session(client, seconds=2.0):
    clock = [0.0]
    return RigSession(client, arm_limits=LIMITS, episode_seconds=seconds,
                      clock=lambda: clock[0], sleep=lambda d: clock.__setitem__(0, clock[0] + max(d, 0)),
                      log=lambda *_: None)


class RigSessionTests(unittest.TestCase):
    def test_reset_returns_to_the_runner_start_condition(self):
        client = SimClient(PARAMS, seed=3)
        s = session(client)
        s.start(confirm=lambda _: "RUN")
        try:
            # Swing the pendulum hard and move the arm off centre.
            episode = s.run_episode(lambda obs: (0.8 if (len(obs) and obs[0] < 0.6) else -0.8, 0.0))
            self.assertEqual(len(episode.rows), 100)
            s.reset()
            for _ in range(5):  # the next episode's start check, several reads in a row
                validate_state(client.get_state(), startup=True)
        finally:
            s.stop()
        self.assertFalse(client.engaged)

    def test_reset_timeout_stops_the_session(self):
        client = SimClient(PARAMS, seed=4)
        s = session(client)
        s.start(confirm=lambda _: "RUN")
        import pendulum_lab.rig_session as rs
        saved = rs.RESET_TIMEOUT_S
        rs.RESET_TIMEOUT_S = 0.1
        try:
            s.run_episode(lambda obs: (0.8 if obs[0] < 0.6 else -0.8, 0.0))
            with self.assertRaises(SessionStop):
                s.reset()
        finally:
            rs.RESET_TIMEOUT_S = saved
            s.stop()
        self.assertFalse(client.engaged)

    def test_assisted_start_hands_over_inside_the_window(self):
        client = SimClient(PARAMS, seed=5)
        s = session(client, seconds=1.0)
        s.start(confirm=lambda _: "RUN")
        start = AssistedStart(target_energy=0.99, theta_rad=math.radians(12), pen_speed=3.0, tolerance=0.03)
        seen = []
        try:
            episode = s.run_episode(lambda obs: (seen.append(obs) or 0.0, 0.0), start=start)
        finally:
            s.stop()
        self.assertEqual(episode.stop_reason, "duration")
        self.assertEqual(len(episode.rows), 50)
        self.assertTrue(start.inside(episode.handover))
        self.assertGreater(len(episode.lift_rows), 50)
        lift_arm = max(abs(r[1]) for r in episode.lift_rows)
        self.assertLess(lift_arm, LIMITS[2])  # inside the shared arm envelope
        # The first observation the policy sees ends with the handover state
        # and the lift's last command.
        self.assertAlmostEqual(seen[0][-1], episode.lift_rows[-1][-1], places=6)

    def test_lift_timeout_gives_no_policy_ticks(self):
        client = SimClient(PARAMS, seed=6)
        s = session(client)
        s.start(confirm=lambda _: "RUN")
        impossible = AssistedStart(target_energy=0.99, theta_rad=math.radians(0.01), pen_speed=0.01,
                                   timeout_s=2.0)
        try:
            episode = s.run_episode(lambda obs: self.fail("policy must not run"), start=impossible)
            self.assertEqual(episode.stop_reason, "lift_timeout")
            self.assertEqual(episode.rows, [])
            s.reset()  # the session can recover with a normal reset
        finally:
            s.stop()

    def test_assisted_start_keeps_the_hanging_start_check(self):
        client = SimClient(PARAMS, seed=7)
        s = session(client)
        s.start(confirm=lambda _: "RUN")
        try:
            s.run_episode(lambda obs: (0.8 if obs[0] < 0.6 else -0.8, 0.0))  # pendulum left swinging
            start = AssistedStart(target_energy=0.99, theta_rad=math.radians(12), pen_speed=3.0)
            with self.assertRaisesRegex(RuntimeError, "Start requires"):
                s.run_episode(lambda obs: (0.0, 0.0), start=start)
        finally:
            s.stop()

    def test_swing_amplitude_combines_angle_and_speed(self):
        # At the bottom of a 2 deg swing the angle is zero but the speed is not.
        speed = math.radians(2.0) * 12.27
        self.assertAlmostEqual(swing_amplitude([0.0, math.pi, 0.0, speed]), math.radians(2.0), places=6)


if __name__ == "__main__":
    unittest.main()
