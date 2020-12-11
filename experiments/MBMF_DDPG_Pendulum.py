import os
import unittest
import numpy as np
from baconian.envs.gym_env import make
from baconian.algo.ddpg import DDPG
from baconian.core.agent import DDPG_Agent
from baconian.envs.cyber_env import PendulumnCyber
from baconian.config.global_config import GlobalConfig
from baconian.algo.policy import DeterministicMLPPolicy
from baconian.common.logging import Logger, ConsoleLogger
from baconian.core.global_var import SinglentonStepCounter
from baconian.common.sampler.sample_data import TransitionData
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.common.noise import AgentActionNoiseWrapper, UniformNoise
from baconian.common.schedules import ConstantScheduler, DDPGNoiseScheduler
from baconian.core.experiment import Experiment
from baconian.core.experiment_runner import single_exp_runner
from baconian.core.flow.train_test_flow import create_train_test_flow

GlobalConfig().set('DEFAULT_LOG_PATH', './MBMF_DDPG_log_path')
GlobalConfig().set('DEFAULT_LOGGING_FORMAT', '%(message)s')


def task_fn():
    env = make('Pendulum-v0')
    name='mb_test'
    env_spec = env.env_spec
    model_path = '/home/yitongx/Documents/baconian-project/experiments/log'
    cyber = PendulumnCyber(env=env, epoch_to_use=60, use_traj_input=False, use_mbmf=True, \
                           model_path=model_path)
    mlp_config = [
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
                              mlp_config=mlp_config)
    mlp_policy = DeterministicMLPPolicy(env_spec=env_spec,
                                        name=name+'_mlp_policy',
                                        name_scope=name+'_mlp_policy',
                                        output_high=env.observation_space.high,
                                        mlp_config=mlp_config,
                                        reuse=False)
    polyak = 0.995
    gamma = 0.99
    noise_scale = 0.5
    noise_decay = 0.999 # default 0.995
    batch_size = 128
    actor_lr = 0.001    # default 0.001
    critic_lr = 0.001   # default 0.001
    buffer_size = 100000
    total_steps = 500000  # default 1000000
    max_step_per_episode = 500  # reset env when counter > max_step_per_episode
    train_after_step = 10000 # default 10000
    train_every_step = 1
    train_iter_per_call = 1
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


    flow = create_train_test_flow(
        env=env,
        cyber=cyber,
        agent=agent,
        num_test=num_test,
        total_steps=total_steps,
        max_step_per_episode=max_step_per_episode,
        train_after_step=train_after_step,
        test_after_step=test_after_step,
        train_every_step=train_every_step,
        test_every_step=test_every_step,
        train_func_and_args=(agent.train, (), dict()),
        test_func_and_args=(agent.test, (), dict()),
        sample_func_and_args=(agent.sample, (), dict()),
        flow_type='DDPG_TrainTestFlow'
    )

    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name
    )
    experiment.run()

if __name__ == '__main__':
    single_exp_runner(task_fn, del_if_log_path_existed=True, keep_session=True)