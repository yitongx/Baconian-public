import unittest
from baconian.test.tests.set_up.setup import TestWithAll
from baconian.envs.gym_env import make
from baconian.envs.gym_env import Env
from baconian.envs.cyber_env import PendulumnCyber

class test_CyberEnv(unittest.TestCase):
    def test_PendulumCyber(self):
        env = make('Pendulum-v0')
        assert isinstance(env, Env)

        cyber = PendulumnCyber(env=env, epoch_to_use=60, use_traj_input=False, use_mbmf=True, \
                               model_path='/home/yitongx/Documents/baconian-project/experiments/log')
        obs = env.reset()
        action = env.action_space.sample()
        self.assertEqual(obs.shape[0], 3)
        self.assertEqual(action.shape[0], 1)
        self.assertEqual(env.action_space, cyber.action_space)

        obs_, reward, done, info = cyber.step(obs=obs, act=action)
        print(obs_)
        action = cyber.action_space.sample()

