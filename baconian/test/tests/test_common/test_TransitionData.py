from baconian.common.sampler.sample_data import TrajectoryData, TransitionData, MPC_TransitionData
from baconian.envs.gym_env import make
import unittest
import numpy as np
from baconian.algo.rl_algo import ModelBasedAlgo
from baconian.algo.dynamics.dynamics_model import DynamicsModel
from baconian.config.dict_config import DictConfig
from baconian.core.parameters import Parameters
from baconian.config.global_config import GlobalConfig
from baconian.common.misc import *
from baconian.algo.policy.policy import Policy
from baconian.common.logging import ConsoleLogger
from baconian.common.logging import record_return_decorator
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.test.tests.set_up.setup import TestWithAll
from baconian.envs.gym_env import ModifiedHalfCheetahEnv, Env
from baconian.common.data_pre_processing import StandardScaler
from baconian.algo.policy import UniformRandomPolicy
from baconian.algo.mpc import ModelPredictiveControl, ModelBasedModelPredictiveControl
from baconian.algo.dynamics.reward_func.reward_func import MBMPC_HalfCheetah_CostFunc
from baconian.core.agent import Agent, MB_MPC_Agent
from baconian.algo.misc import EpsilonGreedy

class test_TransitionData(TestWithAll, unittest.TestCase):



    def _test_1(self):
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        print(local.items())
        env = local['env']
        assert isinstance(env, ModifiedHalfCheetahEnv)
        env_spec = env.env_spec

        batch_data = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                   action_shape=env_spec.action_shape)
        batch_size = 32
        obs = env.reset()
        for i in range(batch_size):
            act = self.RandomController_get_action(env=env, state=obs)
            obs_, reward, done, info = env.step(action=act)
            batch_data.append(obs, act, obs_, done, reward)
        self.assertEqual(len(batch_data), batch_size)

        mlp_dyna.init()
        train_epoch = 20
        for i in range(train_epoch):
            res = mlp_dyna.train(batch_data, train_iter=10)
            print('iter:{} loss:{}'.format(i, res))



    # def MPCActionSample(self, state):
    def cheetah_cost_fn(self, state, action, next_state):
        if len(state.shape) > 1:
            heading_penalty_factor = 10
            scores = np.zeros((state.shape[0],))

            # dont move front shin back so far that you tilt forward
            front_leg = state[:, 5]
            my_range = 0.2
            scores[front_leg >= my_range] += heading_penalty_factor

            front_shin = state[:, 6]
            my_range = 0
            scores[front_shin >= my_range] += heading_penalty_factor

            front_foot = state[:, 7]
            my_range = 0
            scores[front_foot >= my_range] += heading_penalty_factor

            scores -= (next_state[:, 17] - state[:, 17]) / 0.01 + 0.1 * (np.sum(action ** 2, axis=1))
            return scores

        heading_penalty_factor = 10
        score = 0

        # dont move front shin back so far that you tilt forward
        front_leg = state[5]
        my_range = 0.2
        if front_leg >= my_range:
            score += heading_penalty_factor

        front_shin = state[6]
        my_range = 0
        if front_shin >= my_range:
            score += heading_penalty_factor

        front_foot = state[7]
        my_range = 0
        if front_foot >= my_range:
            score += heading_penalty_factor

        score -= (next_state[17] - state[17]) / 0.01 + 0.1 * (np.sum(action ** 2))
        return score

    def trajectory_cost_fn(self, cost_fn, states, actions, next_states):
        trajectory_cost = 0
        for i in range(len(actions)):
            trajectory_cost += cost_fn(states[i], actions[i], next_states[i])

        return trajectory_cost


    def test_random_buffer_1(self):
        env = make('ModifiedHalfCheetah')
        # env.init()
        env_spec = env.env_spec
        random_buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=5)
        rl_buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                   action_shape=env_spec.action_shape, size=10)

        max_step = 10
        ep_len = 0
        obs = env.reset()
        while ep_len < max_step:
            act = self.RandomController_get_action(env=env, state=obs)
            obs_, reward, done, _ = env.step(act)
            random_buffer.append(obs, act, obs_, done, reward)
            assert not done
            obs = obs_
            ep_len += 1

    def test_Transition_union(self):
        '''
        Useless testcase.
        '''
        algo, locals = self.create_mpc(name='test_Transition_union')
        env_spec = locals['env_spec']
        env = locals['env']
        env.env_spec = env_spec

        algo.init()
        for _ in range(100):
            assert env_spec.action_space.contains(algo.predict(env_spec.obs_space.sample()))

        st = env.reset()
        data = TransitionData(env_spec)

        for _ in range(10):
            ac = algo.predict(st)
            new_st, re, done, _ = env.step(action=ac)
            data.append(state=st,
                        new_state=new_st,
                        reward=re,
                        action=ac,
                        done=done)
        print(algo.train(batch_data=data))


    def test_sample_batch_as_Transition_1(self):
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        print(local.items())
        env = local['env']
        assert isinstance(env, ModifiedHalfCheetahEnv)
        env_spec = env.env_spec

        batch_data = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                    action_shape=env_spec.action_shape)
        batch_size = 32
        obs = env.reset()
        for i in range(batch_size):
            act = self.RandomController_get_action(env=env, state=obs)
            obs_, reward, done, info = env.step(action=act)
            batch_data.append(obs, act, obs_, done, reward)
        self.assertEqual(len(batch_data), batch_size)

        mlp_dyna.init()
        train_epoch = 20
        for i in range(train_epoch):
            res = mlp_dyna.train(batch_data, train_iter=10)
            print('iter:{} loss:{}'.format(i, res))


    def test_sample_batch_as_Transition_2(self):
        env = make('ModifiedHalfCheetah')
        env.init()
        env_spec = env.env_spec
        random_buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=100)
        print("====> Random Sample")
        num_trajectory = 1
        max_step = 100

        for i in range(num_trajectory):
            ep_len = 0
            obs = env.reset()
            while ep_len < max_step:
                act = self.RandomController_get_action(env=env, state=obs)
                obs_, reward, done, _ = env.step(act)
                random_buffer.append(obs, act, obs_, done, reward)
                assert not done
                obs = obs_
                ep_len += 1

        batch_data = random_buffer.sample_batch_as_Transition(batch_size=32, shuffle_flag=True, all_as_batch=True)
        self.assertEqual(len(batch_data), 4)
        self.assertEqual(len(batch_data[-1]), 4)

    def test_sample_batch(self):
        env = make('ModifiedHalfCheetah')
        env.init()
        env_spec = env.env_spec
        random_buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=100)
        print("====> Random Sample")
        num_trajectory = 1
        max_step = 30

        for i in range(num_trajectory):
            ep_len = 0
            obs = env.reset()
            while ep_len < max_step:
                act = self.RandomController_get_action(env=env, state=obs)
                obs_, reward, done, _ = env.step(act)
                random_buffer.append(obs, act, obs_, done, reward)
                assert not done
                obs = obs_
                ep_len += 1

        batch_data_1 = random_buffer.sample_batch(batch_size=16, shuffle_flag=True)
        assert isinstance(batch_data_1, dict)
        print(batch_data_1.keys())

        self.assertEqual(len(batch_data_1['action_set']), 16)
        # batch_data_2 = random_buffer.sample_batch(batch_size=32, shuffle_flag=True)

    def test_dagger_1(self):
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        mlp_dyna.init()
        env = local['env']
        assert isinstance(env, ModifiedHalfCheetahEnv)
        env_spec = env.env_spec

        random_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=5)
        rl_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                   action_shape=env_spec.action_shape, size=10)

        obs = np.zeros((18), dtype=np.float32)
        for i in range(5):
            act = np.zeros((6), dtype=np.float32)
            obs_, reward, done, _ = np.zeros((18), dtype=np.float32), 0., False, 0.
            random_buffer.append(obs, act, obs_, done, reward)
            obs = obs_
        dagger_buffer = random_buffer.union(rl_buffer, rand_rl_ratio=0.1)
        self.assertEqual(dagger_buffer, random_buffer)

        obs = env.reset()
        for i in range(10):
            act = self.RandomController_get_action(env=env, state=obs)
            obs_, reward, done, _ = env.step(act)
            rl_buffer.append(obs, act, obs_, done, reward)
            assert not done
            obs = obs_

        dagger_buffer = random_buffer.union(rl_buffer, rand_rl_ratio=0.1)
        self.assertEqual(len(dagger_buffer), 11)



    def RandomController_get_action(self, env, state):
        return env.action_space.sample()



    def random_buffer_sample(self, env, buffer, num_trajectory, max_step):
        '''RandomController.sample()'''
        for i in range(num_trajectory):
            obs = env.reset()
            ep_len = 0
            for j in range(max_step):
                act = self.RandomController_get_action(env, obs)
                obs_, rew, done, _ = env.step(act)
                buffer.append(obs, act, obs_, done, rew)
                if done:
                    break
                else:
                    obs = obs_
        return buffer


    def DynaMLP_get_action(self, mlp_dyna: DynamicsModel, env: Env, state, cost_fn, num_simulated_paths, horizon):
        '''
        mpc.ModelBasedModelPredictiveControl.predict()

        :param mlp_dyna:
        :param env:
        :param state:
        :param cost_fn:
        :param num_simulated_paths:
        :param horizon:
        :return:
        '''
        rollout = TrajectoryData(env_spec=env.env_spec)
        for i in range(num_simulated_paths):
            path = TransitionData(env_spec=env.env_spec)
            obs = state
            for j in range(horizon):
                action = env.action_space.sample()
                obs_ = mlp_dyna.step(action=action, state=obs)
                cost = cost_fn(obs, action, obs_)
                path.append(obs, action, obs_, False, -cost)
                obs = obs_

            rollout.append(path)
        rollout.trajectories.sort(key=lambda x: x.cumulative_reward, reverse=True)
        optimial_action = rollout.trajectories[0].action_set[0]
        return optimial_action


    def rl_buffer_sample(self, mlp_dyna, env, buffer, num_trajectory, max_step, num_simulated_paths, iter):

        for i in range(num_trajectory):
            obs = env.reset()
            ep_len = 0
            for j in range(max_step):
                act = self.DynaMLP_get_action(mlp_dyna, env, obs, self.cheetah_cost_fn, \
                                              num_simulated_paths=num_simulated_paths, horizon=20)
                obs_, rew, done, _ = env.step(act)
                buffer.append(obs, act, obs_, done, rew)
                if done:
                    break
                else:
                    obs = obs_

                if j % 5 == 0:
                    print('iter:{} num_trajectory:{}/{} step:{}/{}'.format(iter, i, num_trajectory-1, j, max_step-1))

        return buffer


    def test_dagger_2(self):
        '''
        MPC Training without normalization
        '''
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        mlp_dyna.init()
        print(mlp_dyna.state_input_scaler)

        env = local['env']
        assert isinstance(env, ModifiedHalfCheetahEnv)
        env_spec = env.env_spec

        num_trajectory = 1  # default 10
        max_step = 50  # default 1000
        on_policy_iter = 10
        batch_size = 16
        num_simulated_paths = 100 # default 1000

        random_buffer_size = 500    # default 1000
        rl_buffer_size = 500   # default 1000
        random_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=max_step)
        rl_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                   action_shape=env_spec.action_shape, size=max_step)
        print("====> Prepare random_buffer")
        random_buffer = self.random_buffer_sample(env, random_buffer, num_trajectory, max_step)

        print("====> Start Training")
        for iter in range(on_policy_iter):
            data = random_buffer.union(rl_buffer, rand_rl_ratio=0.1)
            batch_data_list = data.sample_batch_as_Transition(batch_size=batch_size, shuffle_flag=True, all_as_batch=True)
            for batch_data in batch_data_list:
                print(mlp_dyna.train(batch_data=batch_data, train_iter=10))
            rl_buffer = self.rl_buffer_sample(mlp_dyna, env, rl_buffer, num_trajectory, max_step, num_simulated_paths, iter)


    def test_apply_normalization(self):
        '''
        Test normalization & denormalization in Transition.apply_(de)normalization
        '''
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        mlp_dyna.init()
        print(mlp_dyna.state_input_scaler)

        env = local['env']
        assert isinstance(env, ModifiedHalfCheetahEnv)
        env_spec = env.env_spec
        buffer_size = 50
        random_buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=buffer_size)

        obs = env.reset()
        for i in range(buffer_size):
            act = env.action_space.sample()
            obs_, reward, done, info = env.step(act)
            random_buffer.append(obs, act, obs_, done, reward)

        normalized_random_buffer, mean_dict, var_dict = random_buffer.apply_normalization()
        denormalized_random_buffer = normalized_random_buffer.apply_denormalization(None, mean_dict, var_dict)

        self.assertEqual(random_buffer.action_set.any(), denormalized_random_buffer.action_set.any())
        self.assertEqual(random_buffer.state_set.any(), denormalized_random_buffer.state_set.any())

    def test_dagger_3(self):
        '''
        MPC Training
        Add normalization and denormalization
        '''
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        mlp_dyna.init()
        print(mlp_dyna.state_input_scaler)

        env = local['env']
        assert isinstance(env, ModifiedHalfCheetahEnv)
        env_spec = env.env_spec

        num_trajectory = 1  # default 10
        max_step = 50  # default 1000
        on_policy_iter = 10
        train_iter = 10
        batch_size = 16
        rand_rl_ratio = 0.1
        num_simulated_paths = 100 # default 1000

        random_buffer_size = 500    # default 1000
        rl_buffer_size = 500   # default 1000

        random_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=max_step)
        rl_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                   action_shape=env_spec.action_shape, size=max_step)

        print("====> Prepare random_buffer")
        random_buffer = self.random_buffer_sample(env, random_buffer, num_trajectory, max_step)
        normalized_random_buffer, mean_dict, var_dict = random_buffer.apply_normalization()

        print("====> Start Training")
        for iter in range(on_policy_iter):
            data = normalized_random_buffer.union(rl_buffer, rand_rl_ratio=rand_rl_ratio)
            batch_data_list = data.sample_batch_as_Transition(batch_size=batch_size, shuffle_flag=True, all_as_batch=True)
            for batch_data in batch_data_list:
                print(mlp_dyna.train(batch_data=batch_data, train_iter=train_iter))
            rl_buffer = self.rl_buffer_sample(mlp_dyna, env, rl_buffer, num_trajectory, max_step, num_simulated_paths, iter)



    def rl_buffer_sample_in_algo(self, algo, env, buffer, num_trajectory, max_step, num_simulated_paths, iter):
        for i in range(num_trajectory):
            obs = env.reset()
            ep_len = 0
            for j in range(max_step):
                act = algo.predict(obs, is_reward_func=False)
                obs_, rew, done, _ = env.step(act)
                buffer.append(obs, act, obs_, done, rew)
                if done:
                    break
                else:
                    obs = obs_

                if j % 5 == 0:
                    print('iter:{} num_trajectory:{}/{} step:{}/{}'.format(iter, i, num_trajectory-1, j, max_step-1))

        return buffer

    def test_algo_ModelBasedModelPredictiveControl(self):
        '''
        Test algo = ModelBasedModelPredictiveControl()
        :return:

        '''
        name = 'mbmpc'
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        env = local['env']
        env_spec = env.env_spec

        policy = UniformRandomPolicy(env_spec=env_spec, name='urp')

        algo = ModelBasedModelPredictiveControl(dynamics_model=mlp_dyna,
                                                env_spec=env_spec,
                                                config_or_config_dict=dict(
                                                    SAMPLED_HORIZON=20,
                                                    SAMPLED_PATH_NUM=50,
                                                    dynamics_model_train_iter=10
                                                ),
                                                name=name,
                                                policy=policy)

        algo.set_terminal_reward_function_for_dynamics_env(reward_func=MBMPC_HalfCheetah_CostFunc(name='cost_fn'),
                                                           terminal_func=MBMPC_HalfCheetah_CostFunc(name='terminal_fn'))
        algo.init()


        num_trajectory = 1  # default 10
        max_step = 50  # default 1000
        on_policy_iter = 10
        batch_size = 16
        num_simulated_paths = 100 # default 1000

        random_buffer_size = 500    # default 1000
        rl_buffer_size = 500   # default 1000
        random_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=max_step)
        rl_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                   action_shape=env_spec.action_shape, size=max_step)

        print("====> Prepare random_buffer")
        random_buffer = self.random_buffer_sample(env, random_buffer, num_trajectory, max_step)
        normalized_random_buffer, mean_dict, var_dict = random_buffer.apply_normalization()

        print("====> Start Training")
        for iter in range(on_policy_iter):
            data = normalized_random_buffer.union(rl_buffer, rand_rl_ratio=0.1)
            batch_data_list = data.sample_batch_as_Transition(batch_size=batch_size, shuffle_flag=True, all_as_batch=True)
            for batch_data in batch_data_list:
                print(algo.train(batch_data=batch_data, train_iter=10))
            rl_buffer = self.rl_buffer_sample_in_algo(algo, env, rl_buffer, num_trajectory, max_step, num_simulated_paths, iter)


    def rl_buffer_sample_in_agent(self, agent, env, buffer, num_trajectory, max_step, num_simulated_paths, iter):
        for i in range(num_trajectory):
            obs = env.reset()
            ep_len = 0
            for j in range(max_step):
                act = agent.predict(obs=obs, is_reward_func=False)
                obs_, rew, done, _ = env.step(act)
                buffer.append(obs, act, obs_, done, rew)
                if done:
                    break
                else:
                    obs = obs_

                if j % 5 == 0:
                    print('iter:{} num_trajectory:{}/{} step:{}/{}'.format(iter, i, num_trajectory-1, j, max_step-1))

        return buffer


    def test_agent_MPC(self):
        '''
        Capsule in agent.
        :return:
        '''

        name = 'mb_mpc'
        mlp_dyna, local = self.create_continue_dynamics_model(env_id='ModifiedHalfCheetah', name='mlp_dyna_model')
        env = local['env']
        env_spec = env.env_spec

        policy = UniformRandomPolicy(env_spec=env_spec, name='urp')

        algo = ModelBasedModelPredictiveControl(dynamics_model=mlp_dyna,
                                                env_spec=env_spec,
                                                config_or_config_dict=dict(
                                                    SAMPLED_HORIZON=20,
                                                    SAMPLED_PATH_NUM=50,
                                                    dynamics_model_train_iter=10
                                                ),
                                                name=name,
                                                policy=policy)

        algo.set_terminal_reward_function_for_dynamics_env(reward_func=MBMPC_HalfCheetah_CostFunc(name='cost_fn'),
                                                           terminal_func=MBMPC_HalfCheetah_CostFunc(name='terminal_fn'))
        agent = MB_MPC_Agent(name=name + '_agent',
                             env=env, env_spec=env_spec,
                             algo=algo,
                             exploration_strategy=None,
                             algo_saving_scheduler=None)
        agent.init()

        num_trajectory = 1  # default 10
        max_step = 50  # default 1000
        on_policy_iter = 10
        batch_size = 10
        num_simulated_paths = 100  # default 1000

        # TODO: 9.22 Is there relations between buffer_size and max_step?
        #  Besides, why multiple num_simluated_paths

        random_size = 500  # default 1000
        rl_size = 500  # default 1000
        random_size = max_step
        rl_size = max_step

        random_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                           action_shape=env_spec.action_shape, size=r)
        rl_buffer = MPC_TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                                       action_shape=env_spec.action_shape, size=max_step)

        print("====> Prepare random_buffer")
        random_buffer = self.random_buffer_sample(env, random_buffer, num_trajectory, max_step)
        normalized_random_buffer, mean_dict, var_dict = random_buffer.apply_normalization()

        print("====> Start Training")
        for iter in range(on_policy_iter):
            data = normalized_random_buffer.union(rl_buffer, rand_rl_ratio=0.1)
            batch_data_list = data.sample_batch_as_Transition(batch_size=batch_size, shuffle_flag=True,
                                                              all_as_batch=True)
            for batch_data in batch_data_list:
                agent.train(batch_data=batch_data, train_iter=60)

            rl_buffer = agent.sample(env=env,
                                     sample_count=iter,
                                     buffer=rl_buffer,
                                     num_trajectory=num_trajectory,
                                     max_step=max_step,
                                     num_simulated_paths=num_simulated_paths,
                                     in_which_status='TRAIN',
                                     store_flag=False)
