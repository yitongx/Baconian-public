from baconian.envs.gym_env import GymEnv
from baconian.test.tests.set_up.setup import TestWithLogSet
from gym import make


class TestEnv(TestWithLogSet):
    def test_gym_env(self):
        env = GymEnv('Acrobot-v1')
        env.set_status('TRAIN')
        self.assertEqual(env.total_step_count_fn(), 0)
        self.assertEqual(env._last_reset_point, 0)

        env.init()
        env.seed(10)
        env.reset()
        self.assertEqual(env.total_step_count_fn(), 0)
        self.assertEqual(env._last_reset_point, 0)

        for i in range(1000):
            new_st, re, done, _ = env.step(action=env.action_space.sample())
            self.assertEqual(env.total_step_count_fn(), i + 1)
            if done is True:
                env.reset()
                self.assertEqual(env._last_reset_point, env.total_step_count_fn())
                self.assertEqual(env._last_reset_point, i + 1)
    # def test_all_get_state(self):
    #     type_list = []
    #     for id in GymEnv._all_gym_env_id:
    #         try:
    #             print(id)
    #             env = make(id)
    #             type_list.append(type(env).__name__)
    #             st = env.reset()
    #             self.assertTrue(env.observation_space.contains(st))
    #             assert env.observation_space.contains(st)
    #             del env
    #         except Exception:
    #             print("{} is not found".format(id))
    #         else:
    #             print("{} is found".format(id))
    #     print(set(type_list))
