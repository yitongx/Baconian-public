import os
import unittest
import numpy as np
from baconian.envs.gym_env import make
from baconian.common.sampler.sample_data import TrajectoryData, TransitionData, MPC_TransitionData
from baconian.envs.cyber_env import PendulumnCyber
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.algo.ddpg import DDPG
from baconian.envs.cyber_env import PendulumnCyber
from baconian.core.agent import DDPG_Agent
from baconian.common.noise import AgentActionNoiseWrapper, UniformNoise
from baconian.common.schedules import ConstantScheduler, DDPGNoiseScheduler
from baconian.core.global_var import SinglentonStepCounter
from baconian.common.logging import Logger, ConsoleLogger
from baconian.config.global_config import GlobalConfig
from baconian.test.tests.set_up.setup import TestWithAll

class Test_DDPG(unittest.TestCase):
    def LogSetup(self):
        Logger().init(config_or_config_dict=GlobalConfig().DEFAULT_LOG_CONFIG_DICT,
                      log_path=GlobalConfig().DEFAULT_LOG_PATH,
                      log_level=GlobalConfig().DEFAULT_LOG_LEVEL)
        ConsoleLogger().init(logger_name='console_logger',
                             to_file_flag=True,
                             level=GlobalConfig().DEFAULT_LOG_LEVEL,
                             to_file_name=os.path.join(Logger().log_dir, 'console.log'))

        self.assertTrue(ConsoleLogger().inited_flag)
        self.assertTrue(Logger().inited_flag)

    def test_DDPG_1(self):
        self.LogSetup()
        env = make('Pendulum-v0')
        name='mb_test'
        env_spec = env.env_spec
        cyber = PendulumnCyber(env=env, epoch_to_use=60, use_traj_input=False, use_mbmf=True, \
                               model_path='/home/yitongx/Documents/baconian-project/experiments/log')
        actor_policy_mlp_config = [
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "1",
                                          "N_UNITS": 32,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "2",
                                          "N_UNITS": 16,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "3",
                                          "N_UNITS": 8,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "TANH",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "OUPTUT",
                                          "N_UNITS": 1,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      }
                                  ]
        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name=name+'_mlp_q',
                                  name_scope=name+'_mlp_q',
                                  output_high=env.action_space.high,
                                  mlp_config=actor_policy_mlp_config)
        mlp_policy = DeterministicMLPPolicy(env_spec=env_spec,
                                            name=name+'_mlp_policy',
                                            name_scope=name+'_mlp_policy',
                                            output_high=env.observation_space.high,
                                            mlp_config=actor_policy_mlp_config,
                                            reuse=False)
        polyak = 0.995
        gamma = 0.99
        batch_size = 128
        actor_lr = 0.001
        critic_lr = 0.001
        train_iter_per_call = 1
        buffer_size = 100000
        total_steps = 100000  # default 1000000
        max_step_per_episode = 500  # reset env when counter > max_step_per_episode
        train_after_step = 10000
        test_after_step = 10000
        train_every_step = 1
        test_every_step = 1000
        num_test = 10

        algo = DDPG(
            env_spec=env_spec,
            config_or_config_dict={
                "REPLAY_BUFFER_SIZE": buffer_size,
                "GAMMA": gamma,
                "CRITIC_LEARNING_RATE": critic_lr,
                "ACTOR_LEARNING_RATE": actor_lr,
                "DECAY": polyak,
                "BATCH_SIZE": batch_size,
                "TRAIN_ITERATION": train_iter_per_call,
                "critic_clip_norm": 0.1,
                "actor_clip_norm": 0.1,
            },
            value_func=mlp_q,
            policy=mlp_policy,
            name=name + '_ddpg',
            replay_buffer=None
        )

        algo.init()
        buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, action_shape=env_spec.action_shape)
        for i in range(100):    # num_trajectory
            obs = env.reset()
            for j in range(1000):
                action = env.action_space.sample()
                obs_, reward, done, info = env.step(action)
                buffer.append(obs, action, obs_, done, reward)
                if done:
                    break
                else:
                    obs = obs_
        algo.append_to_memory(buffer)

        for i in range(10):
            res = algo.train()
            print(res)

        obs = env.reset()
        act = env.action_space.sample()
        obs_, reward, done, info = cyber.step(obs, act)
        print('obs_', obs_)


        obs = env.observation_space.sample()
        obs_batch = np.array([])
        for i in range(5):
            obs_batch = np.concatenate((obs_batch, env.observation_space.sample()), axis=0)
            # print(obs_batch.shape)

        act_batch = algo.predict(obs_batch)
        print(act_batch.shape)
        print(act_batch)

        print('====> Test')
        act = algo.predict(obs)
        print(act.shape)
        obs = env.reset()
        print(obs.shape)
        obs_, reward, done, info = cyber.step(obs, act)



    def test_DDPG_2(self):
        self.LogSetup()
        env = make('Pendulum-v0')
        name='mb_test'
        env_spec = env.env_spec
        cyber = PendulumnCyber(env=env, epoch_to_use=60, use_traj_input=False, use_mbmf=True, \
                               model_path='/home/yitongx/Documents/baconian-project/experiments/log')
        actor_policy_mlp_config = [
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "1",
                                          "N_UNITS": 32,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "2",
                                          "N_UNITS": 16,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "3",
                                          "N_UNITS": 8,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "TANH",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "OUPTUT",
                                          "N_UNITS": 1,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      }
                                  ]
        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name=name+'_mlp_q',
                                  name_scope=name+'_mlp_q',
                                  output_high=env.action_space.high,
                                  mlp_config=actor_policy_mlp_config)
        mlp_policy = DeterministicMLPPolicy(env_spec=env_spec,
                                            name=name+'_mlp_policy',
                                            name_scope=name+'_mlp_policy',
                                            output_high=env.observation_space.high,
                                            mlp_config=actor_policy_mlp_config,
                                            reuse=False)
        polyak = 0.995
        gamma = 0.99
        noise_scale = 0.5
        noise_decay = 0.995
        batch_size = 128
        actor_lr = 0.001
        critic_lr = 0.001
        train_iter_per_call = 1
        buffer_size = 100000

        total_steps = 100000  # default 1000000
        max_step_per_episode = 500  # reset env when counter > max_step_per_episode
        train_after_step = 10000 # default 10000
        train_every_step = 1
        test_after_step = 10000
        test_every_step = 1000
        num_test = 10

        algo = DDPG(
            env_spec=env_spec,
            config_or_config_dict={
                "REPLAY_BUFFER_SIZE": buffer_size,
                "GAMMA": gamma,
                "CRITIC_LEARNING_RATE": critic_lr,
                "ACTOR_LEARNING_RATE": actor_lr,
                "DECAY": polyak,
                "BATCH_SIZE": batch_size,
                "TRAIN_ITERATION": train_iter_per_call,
                "critic_clip_norm": 0.1,
                "actor_clip_norm": 0.1,
            },
            value_func=mlp_q,
            policy=mlp_policy,
            name=name + '_ddpg',
            replay_buffer=None
        )

        step_counter = SinglentonStepCounter(-1)
        noise_adder = AgentActionNoiseWrapper(noise=UniformNoise(scale=noise_scale),
                                              action_weight_scheduler=ConstantScheduler(1.),
                                              noise_weight_scheduler=DDPGNoiseScheduler(train_every_step=train_every_step,
                                                                                        train_after_step=train_after_step,
                                                                                        noise_decay=noise_decay,
                                                                                        step_counter=step_counter))
        agent = DDPG_Agent(env=env,
                           algo=algo,
                           env_spec=env_spec,
                           noise_adder=noise_adder,
                           name=name+'_agent')

        agent.init()
        test_reward = []
        data_sample = []
        obs, ep_ret, ep_len = env.reset(), 0, 0
        for step in range(total_steps):
            step_counter.increase(1)
            act = agent.predict(obs=obs)
            obs_, reward, done, _ = cyber.step(obs, act)
            _buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, action_shape=env_spec.action_shape)
            _buffer.append(obs, act, obs_, done, reward)
            agent.algo.append_to_memory(_buffer)

            ep_ret += reward
            ep_len += 1

            if done or ep_len > max_step_per_episode:
                obs, ep_ret, ep_len = env.reset(), 0, 0
            else:
                obs = obs_
            if step > train_after_step and step % train_every_step == 0:
                agent.train()
            if step > test_after_step and step % test_every_step == 0:
                data_sample, test_reward = agent.test(env=env,
                                                      cyber=cyber,
                                                      data_sample=data_sample,
                                                      test_reward=test_reward,
                                                      num_test=num_test,
                                                      max_step_per_episode=max_step_per_episode)


