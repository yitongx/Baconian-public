from baconian.envs.gym_env import make
from baconian.core.core import EnvSpec
from baconian.core.agent import MB_MPC_Agent
from baconian.core.experiment import Experiment
from baconian.algo.policy import UniformRandomPolicy
from baconian.config.global_config import GlobalConfig
from baconian.core.experiment_runner import single_exp_runner
from baconian.algo.mpc import ModelBasedModelPredictiveControl
from baconian.core.flow.train_test_flow import create_train_test_flow
from baconian.algo.dynamics.mlp_dynamics_model import ContinuousMLPGlobalDynamicsModel
from baconian.algo.dynamics.reward_func.reward_func import MBMPC_HalfCheetah_CostFunc
from baconian.algo.dynamics.terminal_func.terminal_func import MBMPC_HalfCheetah_TerminalFunc

GlobalConfig().set('DEFAULT_LOG_PATH', './log_path')
GlobalConfig().set('DEFAULT_LOGGING_FORMAT', '%(message)s')


def task_fn():
    name = 'mpc_ModifiedHalfCheetah'
    env = make('ModifiedHalfCheetah')
    env_spec = env.env_spec

    mlp_dyna = MBMPC_MLPDynamics(
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


    # buffer
    rl_size = 500  # default 1000
    random_size = 500  # default 1000

    ### algo
    horizon = 20
    dyna_epoch = 60

    ### agent
    max_step = 500  # default 1000 # TODO: 9.22 should max_step == rl_size == random_size?
    batch_size = 128
    rand_rl_ratio = 0.1
    random_trajectory = 1    # TODO: 9.22 Is there situations when tranjectory num must != 1
    on_policy_trajectory = 1
    on_policy_iter = 10
    num_simulated_paths = 50  # default 1000

    algo = ModelBasedModelPredictiveControl(
        dynamics_model=mlp_dyna,
        env_spec=env_spec,
        config_or_config_dict=dict(
            SAMPLED_HORIZON=horizon,
            SAMPLED_PATH_NUM=num_simulated_paths,
            dynamics_model_train_iter=dyna_epoch
        ),
        name=name + '_algo',
        policy=UniformRandomPolicy(env_spec=env_spec, name='uniform_random')
    )

    algo.set_terminal_reward_function_for_dynamics_env(reward_func=MBMPC_HalfCheetah_CostFunc(name='cost_fn'),
                                                       terminal_func=MBMPC_HalfCheetah_TerminalFunc(name='terminal_fn'))
    agent = MB_MPC_Agent(name=name + '_agent',
                         env=env, env_spec=env_spec,
                         algo=algo,
                         exploration_strategy=None,
                         algo_saving_scheduler=None)
    flow = create_train_test_flow(
        env = env,
        env_spec = env_spec,
        rl_size = rl_size,
        max_step = max_step,
        batch_size = batch_size,
        random_size = random_size,
        rand_rl_ratio=rand_rl_ratio,
        train_iter=dyna_epoch,
        on_policy_iter=on_policy_iter,
        random_trajectory = random_trajectory,
        on_policy_trajectory = on_policy_trajectory,
        num_simulated_paths = num_simulated_paths,
        train_func_and_args = (agent.train, (), dict()),
        test_func_and_args = (agent.test, (), dict()),
        sample_func_and_args = (agent.sample, (), dict()),
        train_every_sample_count = None,
        test_every_sample_count = None,
        start_train_after_sample_count = None,
        start_test_after_sample_count = None,
        flow_type = 'MBMPC_TrainFlow'
    )



    experiment = Experiment(
        tuner=None,
        env=env,
        agent=agent,
        flow=flow,
        name=name
    )
    experiment.run()

single_exp_runner(task_fn, del_if_log_path_existed=True)
