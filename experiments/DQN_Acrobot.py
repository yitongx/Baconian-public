from baconian.algo.dqn import DQN
from baconian.algo.misc import EpsilonGreedy
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.common.schedules import LinearScheduler
from baconian.config.global_config import GlobalConfig
from baconian.core.agent import Agent
from baconian.core.core import EnvSpec
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import create_train_test_flow
from baconian.core.status import get_global_status_collect
from baconian.envs.gym_env import make

GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
GlobalConfig().set('DEFAULT_LOGGING_FORMAT', '%(message)s')
GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=10000,
                                                                    TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                                                    TOTAL_AGENT_UPDATE_COUNT=None))

def task_fn():
    env = make('Acrobot-v1')
    name = 'demo_exp'
    env.env_spec = EnvSpec(obs_space=env.observation_space,
                       action_space=env.action_space)
    env_spec = env.env_spec
    mlp_q = MLPQValueFunction(env_spec=env_spec,
                              name_scope=name + '_mlp_q',
                              name=name + '_mlp_q',
                              mlp_config=[
                                  {
                                      "ACT": "TANH",
                                      "B_INIT_VALUE": 0.0,
                                      "NAME": "1",
                                      "N_UNITS": 64,
                                      "TYPE": "DENSE",
                                      "W_NORMAL_STDDEV": 0.03
                                  },
                                  {
                                      "ACT": "TANH",
                                      "B_INIT_VALUE": 0.0,
                                      "NAME": "2",
                                      "N_UNITS": 64,
                                      "TYPE": "DENSE",
                                      "W_NORMAL_STDDEV": 0.03
                                  },
                                  {
                                      "ACT": "RELU",
                                      "B_INIT_VALUE": 0.0,
                                      "NAME": "3",
                                      "N_UNITS": 256,
                                      "TYPE": "DENSE",
                                      "W_NORMAL_STDDEV": 0.03
                                  },
                                  {
                                      "ACT": "LINEAR",
                                      "B_INIT_VALUE": 0.0,
                                      "NAME": "OUPTUT",
                                      "N_UNITS": 1,
                                      "TYPE": "DENSE",
                                      "W_NORMAL_STDDEV": 0.03
                                  }
                              ])
    dqn = DQN(env_spec=env_spec,
              config_or_config_dict=dict(REPLAY_BUFFER_SIZE=50000,
                                         GAMMA=0.99,
                                         BATCH_SIZE=32,
                                         LEARNING_RATE=0.001,
                                         TRAIN_ITERATION=1,
                                         DECAY=0),
              name=name + '_dqn',
              value_func=mlp_q)

    epsilon_greedy = EpsilonGreedy(action_space=env_spec.action_space,
                                   prob_scheduler=LinearScheduler(
                                                         t_fn=lambda: get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
                                                         schedule_timesteps=int(0.1 * 100000),
                                                         initial_p=1.0,
                                                         final_p=0.02),
                                                         init_random_prob=0.1)

    agent = Agent(env=env,
                  env_spec=env_spec,
                  algo=dqn,
                  name=name + '_agent',
                  exploration_strategy=epsilon_greedy,
                  noise_adder=None)


    flow = create_train_test_flow(
        test_every_sample_count=1000,
        train_every_sample_count=1,
        start_test_after_sample_count=0,
        start_train_after_sample_count=10000,
        sample_func_and_args=(agent.sample, (), dict(sample_count=1, env=agent.env, store_flag=True)),
        train_func_and_args=(agent.train, (), dict()),
        test_func_and_args=(agent.test, (), dict(sample_count=1)),
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
    from baconian.core.experiment_runner import *
    single_exp_runner(task_fn, del_if_log_path_existed=True)