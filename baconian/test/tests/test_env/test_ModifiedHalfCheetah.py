from baconian.envs.gym_env import Env, GymEnv, make, ModifiedHalfCheetahEnv
from baconian.test.tests.set_up.class_creator import ClassCreatorSetup
import unittest
from baconian.core.status import *
import baconian.common.spaces as garage_space
import gym.spaces as GymSpace
import numpy as np

class TestEnv(ClassCreatorSetup):
    def test_halfcheetah_1(self):
        env = make('ModifiedHalfCheetah')
        assert isinstance(env, Env)
        self.assertEqual(env._last_reset_point, 0)
        self.assertEqual(env.total_step_count_fn(), 0)
        self.assertEqual(env.trajectory_level_step_count, 0)

        env.init()  # there is a step() in init() at mujoco_env,
        self.assertEqual(env._last_reset_point, 0)
        self.assertEqual(env.total_step_count_fn(), 1)
        self.assertEqual(env.trajectory_level_step_count, 1)

        env.reset()
        self.assertEqual(env._last_reset_point, 1)
        self.assertEqual(env.total_step_count_fn(), 0)
        self.assertEqual(env.trajectory_level_step_count, 0)

        env.init()
        self.assertEqual(env._last_reset_point, 1)
        self.assertEqual(env.total_step_count_fn(), 1)
        self.assertEqual(env.trajectory_level_step_count, 1)

        iter = 100
        for i in range(iter):
            env.set_status('TEST')
            action = env.action_space.sample()
            env.step(action=action)
            self.assertEqual(env._last_reset_point, 1)
            self.assertEqual(env.total_step_count_fn(), env.trajectory_level_step_count)
            self.assertEqual(env.total_step_count_fn(), i + 2)

        env.reset()
        self.assertEqual(env._last_reset_point, iter + 1)
        self.assertEqual(env.total_step_count_fn(), 0)
        self.assertEqual(env.trajectory_level_step_count, 0)


    def test_halfcheetah_2(self):
        env = make('ModifiedHalfCheetah')
        assert not hasattr(env.env_spec, 'flat_obs_dim')
        env.init()
        assert hasattr(env.env_spec, 'flat_obs_dim')
        print(env.env_spec.flat_obs_dim)

        obs = env.reset()
        self.assertEqual(obs.shape, (18,))

    def test_halfcheetah_3(self):
        env = make('ModifiedHalfCheetah')
        observation_space = env.observation_space
        print(observation_space)
        obs_shape = list(observation_space.shape)
        print(obs_shape) # [18]
        '''
        num_simulated_paths = 100
        horizon = 20
        sampled_acts = np.array(
            [[env.action_space.sample() for _ in range(num_simulated_paths)] for _ in range(horizon)]
        )
        # print(sampled_acts)
        self.assertEqual(sampled_acts.shape, (horizon, num_simulated_paths, env.action_space.shape[0]))
        '''