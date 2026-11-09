import unittest

from pendulum_lab.plant.actuator_transport import DelayedAccelerationPlanner


def run(kick, dt, steps):
    p = DelayedAccelerationPlanner(delay_s=0.004, speed_limit_rad_s=5.0, zero_cross_kick_rad_s=kick)
    p.reset(velocity_rad_s=-0.3, active_accel_rad_s2=-50.0)
    # +50 activates partway through the period, after which velocity crosses zero.
    p.command(50.0)
    for _ in range(steps):
        p.advance(dt)
    return p


class ZeroCrossKickTests(unittest.TestCase):
    def test_result_does_not_depend_on_step_size(self):
        coarse, fine = run(0.3, 0.020, 1), run(0.3, 0.001, 20)
        self.assertAlmostEqual(coarse.velocity_rad_s, fine.velocity_rad_s, places=9)
        self.assertAlmostEqual(coarse.position_rad, fine.position_rad, places=9)

    def test_one_kick_per_crossing(self):
        ideal, kicked = run(0.0, 0.001, 20), run(0.3, 0.001, 20)
        self.assertAlmostEqual(ideal.velocity_rad_s, 0.3, places=9)  # crossed zero once
        self.assertAlmostEqual(kicked.velocity_rad_s - ideal.velocity_rad_s, 0.3, places=9)

    def test_disabled_by_default(self):
        p = DelayedAccelerationPlanner(delay_s=0.0, speed_limit_rad_s=5.0)
        p.reset(velocity_rad_s=-0.1, active_accel_rad_s2=50.0)
        p.advance(0.02)
        self.assertAlmostEqual(p.velocity_rad_s, 0.9)


if __name__ == "__main__":
    unittest.main()
