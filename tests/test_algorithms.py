"""Controller contracts, linearization, and short real RL training round trips."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pendulum_lab.contracts import PARAMS, PROFILE, observation_frame, state_from_obs
from pendulum_lab.environment import make_env


class ControllerTests(unittest.TestCase):
    def test_mpc_split_delay_uses_queued_commands_and_checks_arm_speed_at_capture(self):
        from pendulum_lab.algorithms.lqr import Controller as LQR
        from pendulum_lab.algorithms.mpc import Controller as MPC

        config = {"delay": 0.015, "pendulum_input_delay_s": 0.025,
                  "capture_arm_speed": 3.0}
        c = MPC(params_path=PARAMS, config=config)
        lqr = LQR(params_path=PARAMS, config={"delay": 0.015, "pendulum_input_delay_s": 0.025})
        np.testing.assert_allclose(c.A, lqr.A)
        np.testing.assert_allclose(c.B, lqr.B)

        def observe(arm_speed):
            return np.concatenate([observation_frame([0, 0, arm_speed, 0], u)
                                   for u in (0.0, 0.0, -0.4, 0.6)])

        with patch.object(c, "balance", return_value=0.25) as balance:
            c.act(observe(4.28))
            self.assertFalse(c.balancing)
            balance.assert_not_called()
            self.assertEqual(c.act(observe(0.5)), 0.25)
            np.testing.assert_allclose(balance.call_args.args[0], [0, 0, 0.5, 0, 0.6, -0.4])

    def test_arm_limited_controller_uses_sent_action_history_and_logs_policy_request(self):
        from pendulum_lab.arm_limits import ArmLimitedController
        from unittest.mock import Mock

        policy = Mock()
        policy.act.return_value = 1.0
        controller = ArmLimitedController(policy, delay_s=0.015, speed_rad_s=4.7, travel_rad=1.42)
        frame = observation_frame([0.0, 0.0, 4.2, 0.0], 0.8)
        self.assertLess(controller.act(np.tile(frame, PROFILE.history)), 0.0)
        self.assertEqual(controller.pre_arm_limit_action, 1.0)
        controller.reset()
        self.assertIsNone(controller.requested_action)
        policy.reset.assert_called_once()

    def test_pid_uses_learned_swing_and_keeps_its_own_balance(self):
        from pendulum_lab.algorithms.pid import Controller

        config = {"swing_up": "sac", "balance_state_lead_s": 0.012}
        with patch("pendulum_lab.algorithms.sac.Controller") as swing:
            swing.return_value.act.return_value = 0.4
            c = Controller(params_path=PARAMS, config=config, checkpoint=Path("policy.zip"))
        hanging = np.tile(observation_frame([0, np.pi, 0, 0], 0), PROFILE.history)
        self.assertEqual(c.act(hanging), 0.4)
        upright = np.tile(observation_frame([0, 0, 0, 0], 0.5), PROFILE.history)
        with patch.object(c, "balance", return_value=-0.2) as balance:
            self.assertEqual(c.act(upright), -0.2)
            self.assertGreater(balance.call_args.args[0][3], 0.0)
        swing.return_value.act.assert_called_once()

    def test_lqr_blends_energy_action_only_during_learned_swing(self):
        from pendulum_lab.algorithms.lqr import Controller

        config = {"swing_up": "sac", "swing_energy_mix": 0.2}
        with patch("pendulum_lab.algorithms.sac.Controller") as swing, \
                patch("pendulum_lab.algorithms.lqr.swing_action", return_value=-0.5) as energy:
            swing.return_value.act.return_value = 0.5
            c = Controller(params_path=PARAMS, config=config, checkpoint=Path("policy.zip"))
            hanging = np.tile(observation_frame([0, np.pi, 0, 0], 0), PROFILE.history)
            self.assertAlmostEqual(c.act(hanging), 0.3)
            upright = np.tile(observation_frame([0, 0, 0, 0], 0), PROFILE.history)
            self.assertEqual(c.act(upright), 0.0)
            self.assertTrue(c.balancing)
            energy.assert_called_once()
            swing.return_value.act.assert_called_once()

    def test_lqr_learned_swing_up_hands_off_to_lqr(self):
        from pendulum_lab.algorithms.lqr import Controller
        with patch("pendulum_lab.algorithms.sac.Controller") as swing:
            swing.return_value.act.return_value = 1.0
            c = Controller(params_path=PARAMS, config={"swing_up": "sac"},
                           checkpoint=Path("policy.zip"))
        hanging = np.tile(observation_frame([0, np.pi, 0, 0], 0), PROFILE.history)
        self.assertEqual(c.act(hanging), 1.0)
        upright = np.tile(observation_frame([0, 0, 0, 0], 0), PROFILE.history)
        self.assertEqual(c.act(upright), 0.0)
        self.assertTrue(c.balancing)
        self.assertEqual(swing.return_value.act.call_count, 1)
        rail = np.tile(observation_frame([0, np.pi, 5.0, 0], 1.0), PROFILE.history)
        self.assertLess(c.act(rail), 0.0)
        self.assertFalse(c.balancing)

    def test_lqr_brakes_fast_approaches_in_both_directions(self):
        from pendulum_lab.algorithms.lqr import Controller
        config = {"swing_up": "sac", "approach_brake_speed_rad_s": 4.5,
                  "capture_speed": 6.0, "capture_max_action": 1.0}
        with patch("pendulum_lab.algorithms.sac.Controller") as swing:
            swing.return_value.act.return_value = 0.8
            c = Controller(params_path=PARAMS, config=config, checkpoint=Path("policy.zip"))

        def observation(theta, speed, previous=0.0):
            return np.tile(observation_frame([0, theta, 0, speed], previous), PROFILE.history)

        self.assertLess(c.act(observation(-0.5, 9.0, 0.5)), 0.0)
        self.assertEqual(c.requested_action, 0.8)
        self.assertLess(c.pre_arm_limit_action, 0.0)
        c.reset()
        self.assertGreater(c.act(observation(0.5, -9.0, -0.5)), 0.0)
        c.reset()
        self.assertEqual(c.act(observation(-0.5, -2.0)), 0.8)
        self.assertEqual(c.act(observation(0.0, 0.0)), 0.0)
        self.assertTrue(c.balancing)

    def test_lqr_recenters_only_with_arm_speed_headroom(self):
        from pendulum_lab.algorithms.lqr import Controller
        c = Controller(params_path=PARAMS, config={"arm_recenter_rate_rad_s": 0.35})

        def observe(arm, theta, velocity):
            return c.act(np.tile(observation_frame([arm, theta, velocity, 0], 0), PROFILE.history))

        observe(-0.7, 0.0, 0.0)
        self.assertTrue(c.balancing)
        first = -0.64 + 0.35 * PROFILE.dt
        self.assertAlmostEqual(c.arm_reference_rad, first)
        observe(-0.5, 0.0, 3.7)
        self.assertAlmostEqual(c.arm_reference_rad, first)
        observe(-0.5, 0.0, 3.5)
        self.assertAlmostEqual(c.arm_reference_rad, first + 0.35 * PROFILE.dt)
        observe(-0.5, 0.7, 0.0)
        self.assertFalse(c.balancing)
        self.assertEqual(c.arm_reference_rad, 0.0)

    def test_lqr_observer_uses_device_intervals_across_wrap(self):
        from pendulum_lab.algorithms.lqr import Controller
        c = Controller(params_path=PARAMS, config={"balance_observer": True})
        elapsed_us = np.array([0, 20000, 41000, 60000, 79000, 100000, 121000])
        root = (np.sqrt(c.damping**2 + 4 * c.a) - c.damping) / 2
        for tick, elapsed in enumerate(elapsed_us):
            theta = 0.05 * np.exp(root * (elapsed - elapsed_us[-1]) * 1e-6)
            true_rate = root * theta
            c.set_sample_time_us((2**32 - 40000 + int(elapsed)) % (2**32))
            c.act(np.tile(observation_frame([0, theta, 0, true_rate + 0.3], 0), PROFILE.history))
            if tick < len(elapsed_us) - 1:
                self.assertIsNone(c.estimated_pendulum_velocity_rad_s)
        self.assertLess(abs(c.estimated_pendulum_velocity_rad_s - true_rate), 0.2)
        self.assertAlmostEqual(c.observer_time, PROFILE.dt + elapsed_us[-1] * 1e-6)
        c.reset()
        self.assertIsNone(c.estimated_pendulum_velocity_rad_s)

    def test_lqr_closed_loop_is_stable(self):
        from pendulum_lab.algorithms.lqr import Controller
        c = Controller(params_path=PARAMS, config={})
        self.assertLess(max(abs(np.linalg.eigvals(c.A - c.B @ c.K))), 1)

    def test_lqr_separate_input_delays_share_command_history(self):
        from pendulum_lab.algorithms.lqr import balance_model

        dynamics = np.zeros((4, 4))
        dynamics[0, 2] = dynamics[1, 3] = 1.0
        inputs = np.array([[0.0], [0.0], [50.0], [50.0]])
        transition, command, history = balance_model(dynamics, inputs, 0.010, 0.030)
        self.assertEqual(history, 2)

        first = command[:, 0]
        np.testing.assert_allclose(first, [0.0025, 0.0, 0.5, 0.0, 1.0, 0.0])
        second = transition @ first
        np.testing.assert_allclose(second, [0.0200, 0.0025, 1.0, 0.5, 0.0, 1.0])
        third = transition @ second
        np.testing.assert_allclose(third, [0.0400, 0.0200, 1.0, 1.0, 0.0, 0.0])

    def test_lqr_coupling_calibration_changes_only_pendulum_input_response(self):
        from pendulum_lab.algorithms.lqr import Controller

        config = {"delay": 0.015, "pendulum_input_delay_s": 0.027}
        nominal = Controller(params_path=PARAMS, config=config)
        calibrated = Controller(params_path=PARAMS, config={**config, "balance_coupling_scale": 1.1})
        np.testing.assert_allclose(calibrated.A[[0, 2]], nominal.A[[0, 2]])
        np.testing.assert_allclose(calibrated.B, nominal.B)
        np.testing.assert_allclose(calibrated.A[1:4:2, 4:], 1.1 * nominal.A[1:4:2, 4:])
        self.assertLess(max(abs(np.linalg.eigvals(calibrated.A - calibrated.B @ calibrated.K))), 1)

    def test_mpc_capture_r_checks_entry_with_reference_gains(self):
        from pendulum_lab.algorithms.mpc import Controller as MPC
        base = {"delay": 0.015, "pendulum_input_delay_s": 0.025, "capture_max_action": 1.0}
        reference = MPC(params_path=PARAMS, config={**base, "r": 30.0})
        low_r = MPC(params_path=PARAMS, config={**base, "r": 3.0})
        gated = MPC(params_path=PARAMS, config={**base, "r": 3.0, "capture_r": 30.0})
        np.testing.assert_allclose(gated.K, low_r.K)
        np.testing.assert_allclose(gated.K_capture, reference.K)
        # A lean the r = 30 gate accepts but the r = 3 first command would not.
        lean = np.tile(observation_frame([0, 0.06, 0, 0], 0), 4)
        x = state_from_obs(lean)
        self.assertLessEqual(abs(reference.capture_command(x, lean)), 1.0)
        self.assertGreater(abs(low_r.capture_command(x, lean)), 1.0)
        low_r.act(lean)
        gated.act(lean)
        self.assertFalse(low_r.balancing)
        self.assertTrue(gated.balancing)
        with self.assertRaises(ValueError):
            MPC(params_path=PARAMS, config={**base, "capture_r": 0.0})

    def test_lqr_capture_limit_only_blocks_entry(self):
        from pendulum_lab.algorithms.lqr import Controller
        c = Controller(params_path=PARAMS, config={"capture_max_action": 1.0})
        saturated = np.tile(observation_frame([0, 0.30, 0, 0], 0), 4)
        capturable = np.tile(observation_frame([0, 0.01, 0, 0], 0), 4)

        c.act(saturated)
        self.assertFalse(c.balancing)
        c.act(capturable)
        self.assertTrue(c.balancing)
        self.assertEqual(c.act(saturated), -1.0)
        self.assertTrue(c.balancing)

    def test_lqr_velocity_blend_unwraps_angle_history(self):
        from pendulum_lab.algorithms.lqr import Controller
        c = Controller(params_path=PARAMS, config={"balance_velocity_blend": 0.45})
        baseline = Controller(params_path=PARAMS, config={})
        angles = np.pi + np.array([-2, -1, 0, 1]) * PROFILE.dt * 1.5
        obs = np.concatenate([observation_frame([0, theta, 0, 0.4], 0) for theta in angles])
        x = state_from_obs(obs)
        expected = x.copy()
        expected[3] = 0.55 * x[3] + 0.45 * 1.5
        self.assertAlmostEqual(c.balance(x, obs), baseline.balance(expected, obs), places=5)

    def test_lqr_adapts_once_per_excursion_and_resets(self):
        from pendulum_lab.algorithms.lqr import Controller
        config = {"swing_energy_max_ratio": 1.18, "swing_target_energy_ratio": 1.02,
                  "capture_max_action": 1.0}
        c = Controller(params_path=PARAMS, config=config)

        def observe(theta, speed=0.0):
            c.act(np.tile(observation_frame([0, theta, 0, speed], 0), 4))

        observe(-0.42)
        observe(-1.3)
        self.assertEqual(c.swing_config["swing_target_energy_ratio"], 1.02)
        observe(-1.5)
        raised = c.swing_config["swing_target_energy_ratio"]
        self.assertAlmostEqual(raised, 1.02 + 1.0 - np.cos(0.42))
        observe(-1.6)
        self.assertEqual(c.swing_config["swing_target_energy_ratio"], raised)
        observe(0.5)
        observe(1.6)
        self.assertEqual(c.swing_config["swing_target_energy_ratio"], 1.18)
        observe(0.1, 8.0)
        observe(1.5)
        self.assertLess(c.swing_config["swing_target_energy_ratio"], 1.18)
        self.assertEqual(config["swing_target_energy_ratio"], 1.02)
        c.reset()
        self.assertEqual(c.swing_config["swing_target_energy_ratio"], 1.02)
        self.assertIsNone(c.swing_peak)

    def test_reset_reproduces_action_and_upright_is_equilibrium(self):
        for name in ('pid', 'lqr', 'mpc', 'ilqr'):
            with self.subTest(algorithm=name):
                c = importlib.import_module('pendulum_lab.algorithms.' + name).Controller(
                    params_path=PARAMS, config={})
                upright = np.tile(observation_frame([0, 0, 0, 0], 0), 4)
                self.assertAlmostEqual(c.act(upright), 0, places=6)
                hanging = np.tile(observation_frame([0, np.pi, 0, 0], 0), 4)
                c.reset()
                first = c.act(hanging)
                self.assertNotEqual(first, 0)
                for _ in range(30):
                    c.act(hanging)
                c.reset()
                self.assertEqual(first, c.act(hanging))

    def test_nonlinear_model_matches_analytic_linearization(self):
        from pendulum_lab.algorithms.ilqr import Controller
        c = Controller(params_path=PARAMS, config={})
        from pendulum_lab.contracts import PROFILE
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                      [0, c.a, 0, -c.damping]])
        B = np.array([0, 0, 1, c.b]) * PROFILE.max_accel
        dt = PROFILE.dt
        x = np.array([0.01, -0.001, 0.02, -0.003])
        expected = (np.eye(4) + dt * A + 0.5 * dt**2 * A @ A) @ x
        expected += (dt * B + 0.5 * dt**2 * A @ B) * 0.001
        np.testing.assert_allclose(c.dynamics(x, 0.001), expected, atol=1e-8)

    def test_lqr_completes_hanging_start_full_task(self):
        from pendulum_lab.algorithms.lqr import Controller
        from pendulum_lab.environment import true_state
        from pendulum_lab.evaluation import summarize
        env = make_env(PARAMS)
        try:
            controller = Controller(params_path=PARAMS, config={})
            obs, _ = env.reset(seed=0)
            rows = []
            for tick in range(750):
                action = controller.act(obs)
                obs, _, terminated, truncated, _ = env.step(np.array([action]))
                rows.append([(tick + 1) / 50, *true_state(env), action])
                if terminated or truncated:
                    break
            self.assertFalse(terminated)
            self.assertTrue(summarize(rows, "duration", 15)["success"])
        finally:
            env.close()

    def test_dqn_action_grid_validation(self):
        from pendulum_lab.algorithms.dqn import action_grid
        for invalid in ([0], [0, 2], [0, np.nan], [[0, 1]]):
            with self.assertRaises(ValueError):
                action_grid(invalid)

    def test_sac_resume_preserves_step_count_and_restores_replay(self):
        try:
            from pendulum_lab.algorithms.sac import train
            from stable_baselines3 import SAC
        except ImportError:
            self.skipTest('install the rl optional dependencies')
        import json
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = lambda: make_env(PARAMS, duration=0.2)
            train(env_factory=factory, config={
                'total_steps': 16, 'save_replay_buffer': True,
                'sac': {'learning_starts': 4, 'buffer_size': 100, 'batch_size': 4,
                        'train_freq': 1, 'gradient_steps': 1,
                        'policy_kwargs': {'net_arch': [16, 16]}}}, output_dir=root/'first')
            train(env_factory=factory, config={
                'total_steps': 16, 'resume_checkpoint': str(root/'first/policy.zip'),
                'resume_replay_buffer': str(root/'first/replay_buffer.pkl'),
                'save_replay_buffer': True}, output_dir=root/'next')
            metadata = json.loads((root/'next/continuation.json').read_text())
            self.assertEqual(metadata['initial_steps'], 16)
            self.assertEqual(metadata['final_steps'], 32)
            self.assertTrue(metadata['replay_restored'])
            model = SAC.load(root/'next/policy.zip')
            model.load_replay_buffer(root/'next/replay_buffer.pkl')
            self.assertEqual(model.replay_buffer.size(), 32)
            train(env_factory=factory, config={
                'total_steps': 8, 'resume_checkpoint': str(root/'next/policy.zip'),
                'resume_warmup_steps': 8}, output_dir=root/'warm')
            warm = SAC.load(root/'warm/policy.zip')
            self.assertEqual(warm.num_timesteps, 40)
            self.assertEqual(warm.learning_starts, 40)
            self.assertEqual(warm._n_updates, model._n_updates)

    def test_rl_training_checkpoint_round_trip(self):
        try:
            import stable_baselines3
        except ImportError:
            self.skipTest('install the rl optional dependencies')
        import torch
        torch.set_num_threads(1)
        for name in ('ppo', 'sac', 'dqn'):
            with self.subTest(algorithm=name), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module('pendulum_lab.algorithms.' + name)
                settings = {'policy_kwargs': {'net_arch': [16, 16]}}
                if name == 'ppo':
                    settings.update(n_steps=16, batch_size=8, n_epochs=1)
                else:
                    settings.update(learning_starts=4, buffer_size=100, batch_size=4,
                                    train_freq=1, gradient_steps=1)
                config = {'total_steps': 32, 'seed': 7, 'checkpoint_every': 16, 'normalize_observations': True, 'reward_scale': 0.1, name: settings}
                module.train(env_factory=lambda: make_env(PARAMS, duration=0.2),
                             config=config, output_dir=Path(directory))
                checkpoint = Path(directory) / 'policy.zip'
                c = module.Controller(params_path=PARAMS, config={}, checkpoint=checkpoint)
                obs = np.tile(observation_frame([0, np.pi, 0, 0], 0), 4)
                self.assertLessEqual(abs(c.act(obs)), 1)
                self.assertEqual(c.act(obs), c.act(obs))
                import torch
                expected = torch.tensor(obs) / torch.tensor([1.75, 1, 1, 5, 15, 1] * 4)
                policy = c.policy.policy
                extractor = (policy.features_extractor if name == 'ppo' else
                             policy.actor.features_extractor if name == 'sac' else
                             policy.q_net.features_extractor)
                actual = extractor(torch.tensor(obs).unsqueeze(0))
                np.testing.assert_allclose(actual.detach().cpu().numpy()[0], expected.numpy())
                self.assertTrue((Path(directory) / 'learning_curve.monitor.csv').exists())
                saved = list((Path(directory) / 'checkpoints').glob('*.zip'))
                self.assertTrue(saved)
                other = module.Controller(params_path=PARAMS, config={}, checkpoint=saved[0])
                self.assertLessEqual(abs(other.act(obs)), 1)
