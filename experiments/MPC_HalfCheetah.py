from baconian.core.core import EnvSpec
from baconian.envs.gym_env import make
from baconian.core.agent import Agent
from baconian.algo.misc import EpsilonGreedy
from baconian.core.experiment import Experiment
from baconian.core.flow.train_test_flow import create_train_test_flow
from baconian.algo.mpc import ModelPredictiveControl
from baconian.algo.dynamics.terminal_func.terminal_func import RandomTerminalFunc
from baconian.algo.dynamics.reward_func.reward_func import RandomRewardFunc
from baconian.algo.policy import UniformRandomPolicy
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.config.global_config import GlobalConfig

GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
GlobalConfig().set('DEFAULT_LOGGING_FORMAT', '%(message)s')


def task_fn():
    env = make('HalfCheetah-v2')
    name = 'MPC_HalfCheetah'
    env.env_spec = EnvSpec(obs_space=env.observation_space, action_space=env.action_space)
    env_spec = env.env_spec


    mlp_dyna = ContinuousMLPGlobalDynamicsModel(
        env_spec=env.env_spec,
        name_scope=name + '_mlp_dyna',
        name=name + '_mlp_dyna',
        learning_rate=1e-3,
        mlp_config=[
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "1",
                "L1_NORM": 0.0,
                "L2_NORM": 0.0,
                "N_UNITS": 128,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "TANH",
                "B_INIT_VALUE": 0.0,
                "NAME": "2",
                "L1_NORM": 0.0,
                "L2_NORM": 0.0,
                "N_UNITS": 64,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            },
            {
                "ACT": "LINEAR",
                "B_INIT_VALUE": 0.0,
                "NAME": "OUPTUT",
                "L1_NORM": 0.0,
                "L2_NORM": 0.0,
                "N_UNITS": env_spec.flat_obs_dim,
                "TYPE": "DENSE",
                "W_NORMAL_STDDEV": 0.03
            }
        ])

    algo = ModelPredictiveControl(
        dynamics_model=mlp_dyna,
        env_spec=env_spec,
        config_or_config_dict=dict(
            SAMPLED_HORIZON=20,
            SAMPLED_PATH_NUM=1000,
            dynamics_model_train_iter=60
        ),
        name=name + '_mpc',
        policy=UniformRandomPolicy(env_spec=env_spec, name='uni_policy')
    )

    algo.set_terminal_reward_function_for_dynamics_env(reward_func=RandomRewardFunc(name='reward_func'),
                                                       terminal_func=RandomTerminalFunc(name='random_terminal'), )
    agent = Agent(env=env, env_spec=env_spec,
                  algo=algo,
                  name=name + '_agent',
                  exploration_strategy=EpsilonGreedy(action_space=env_spec.action_space, init_random_prob=0.5))

    flow = create_train_test_flow(
        test_every_sample_count=10,
        train_every_sample_count=10,
        start_test_after_sample_count=5,
        start_train_after_sample_count=5,
        train_func_and_args=(agent.train, (), dict()),
        test_func_and_args=(agent.test, (), dict(sample_count=5)),
        sample_func_and_args=(agent.sample, (), dict(sample_count=5, env=agent.env, store_flag=True))
    )

    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name
    )
    experiment.run()


from baconian.core.experiment_runner import single_exp_runner
single_exp_runner(task_fn, del_if_log_path_existed=True)
