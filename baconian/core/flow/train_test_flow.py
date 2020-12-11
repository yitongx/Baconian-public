import abc
from baconian.config.global_config import GlobalConfig
from baconian.common.logging import ConsoleLogger
from baconian.config.dict_config import DictConfig
from baconian.common.misc import *
from baconian.core.parameters import Parameters
from baconian.core.status import *
from baconian.common.error import *


class Flow(object):
    """
    Interface of experiment flow module, it defines the workflow of the reinforcement learning experiments.
    """
    required_func = ()
    required_key_dict = dict()

    def __init__(self, func_dict):
        """
        Constructor for Flow.

        :param func_dict: the function and its arguments that will be called in the Flow
        :type func_dict: dict
        """
        self.func_dict = func_dict
        for key in self.required_func:
            if key not in func_dict:
                raise MissedConfigError('miss key {}'.format(key))

    def launch(self) -> bool:
        """
        Launch the flow until it finished or catch a system-allowed errors (e.g., out of GPU memory, to ensure the log will be saved safely).

        :return: True if the flow correctly executed and finished
        :rtype: bool
        """
        try:
            return self._launch()
        except GlobalConfig().DEFAULT_ALLOWED_EXCEPTION_OR_ERROR_LIST as e:
            ConsoleLogger().print('error', 'error {} occurred'.format(e))
            return False

    def _launch(self) -> bool:
        """
        Abstract method to be implemented by subclass for a certain workflow.

        :return: True if the flow correctly executed and finished
        :rtype: bool
        """
        raise NotImplementedError

    def _call_func(self, key, **extra_kwargs):
        """
        Call a function that is pre-defined in self.func_dict

        :param key: name of the function, e.g., train, test, sample.
        :type key: str
        :param extra_kwargs: some extra kwargs you may want to be passed in the function calling
        :return: actual return value of the called function if self.func_dict has such function otherwise None.
        :rtype:
        """

        if self.func_dict[key]:
            return self.func_dict[key]['func'](*self.func_dict[key]['args'],
                                               **extra_kwargs,
                                               **self.func_dict[key]['kwargs'])
        else:
            return None


class TrainTestFlow(Flow):
    """
    A typical sampling-trainning and testing workflow, that used by most of model-free/model-based reinforcement
    learning method. Typically, it repeat the sampling(saving to memory if off policy)->training(from memory if
    off-policy, from samples if on-policy)->test
    """
    required_func = ('train', 'test', 'sample')
    required_key_dict = {
        "TEST_EVERY_SAMPLE_COUNT": 1000,
        "TRAIN_EVERY_SAMPLE_COUNT": 1000,
        "START_TRAIN_AFTER_SAMPLE_COUNT": 1,
        "START_TEST_AFTER_SAMPLE_COUNT": 1,
    }

    def __init__(self,
                 train_sample_count_func,
                 config_or_config_dict: (DictConfig, dict),
                 func_dict: dict,
                 ):
        """
        Constructor of TrainTestFlow

        :param train_sample_count_func: a function indicates how much training samples the agent has collected currently. When reach preset value, programm will quit training.
        :type train_sample_count_func: method
        :param config_or_config_dict: a Config or a dict should have the keys: (TEST_EVERY_SAMPLE_COUNT, TRAIN_EVERY_SAMPLE_COUNT, START_TRAIN_AFTER_SAMPLE_COUNT, START_TEST_AFTER_SAMPLE_COUNT)
        :type config_or_config_dict: Config or dict
        :param func_dict: function dict, holds the keys: 'sample', 'train', 'test'. each item in the dict as also should be a dict, holds the keys 'func', 'args', 'kwargs'
        :type func_dict: dict
        """
        super(TrainTestFlow, self).__init__(func_dict=func_dict)
        config = construct_dict_config(config_or_config_dict, obj=self)
        self.parameters = Parameters(source_config=config, parameters=dict())  # hyper parameter instance
        self.time_step_func = train_sample_count_func
        self.last_train_point = -1
        self.last_test_point = -1
        assert callable(train_sample_count_func)    # return TOTAL_AGENT_TRAIN_SAMPLE_COUNT

    def _launch(self) -> bool:
        """
        Launch the flow until it finished or catch a system-allowed errors
        (e.g., out of GPU memory, to ensure the log will be saved safely).

        :return: True if the flow correctly executed and finished
        :rtype: bool
        """
        while True:
            self._call_func('sample')   # algo.sample()

            if self.time_step_func() - self.parameters('TRAIN_EVERY_SAMPLE_COUNT') >= self.last_train_point \
                    and self.time_step_func() > self.parameters('START_TRAIN_AFTER_SAMPLE_COUNT'):
                self.last_train_point = self.time_step_func()
                self._call_func('train')

            if self.time_step_func() - self.parameters('TEST_EVERY_SAMPLE_COUNT') >= self.last_test_point \
                    and self.time_step_func() > self.parameters('START_TEST_AFTER_SAMPLE_COUNT'):
                self.last_test_point = self.time_step_func()
                self._call_func('test')

            if self._is_ended() is True:
                break
        return True

    def _is_ended(self):
        """
        :return: True if an experiment is ended
        :rtype: bool
        """
        key_founded_flag = False
        finished_flag = False
        for key in GlobalConfig().DEFAULT_EXPERIMENT_END_POINT:
            if GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[key] is not None:
                key_founded_flag = True
                if get_global_status_collect()(key) >= GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[key]:
                    ConsoleLogger().print('info',
                                          'pipeline ended because {}: {} >= end point value {}'.
                                          format(key, get_global_status_collect()(key),
                                                 GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[key]))
                    finished_flag = True
        if key_founded_flag is False:
            ConsoleLogger().print(
                'warning',
                '{} in experiment_end_point is not registered with global status collector: {}, experiment may not end'.
                    format(GlobalConfig().DEFAULT_EXPERIMENT_END_POINT, list(get_global_status_collect()().keys())))
        return finished_flag



class MBMPC_TrainFlow(Flow):
    '''
    TrainFlow for MBMPC.
    '''
    required_func = ('train', 'sample')
    required_key_dict = {
        "TRAIN_EVERY_SAMPLE_COUNT": 1,
        "TEST_EVERY_SAMPLE_COUNT": None,
        "START_TRAIN_AFTER_SAMPLE_COUNT": None,
        "START_TEST_AFTER_SAMPLE_COUNT": None,
    }

    def __init__(self,
                 train_sample_count_func,
                 config_or_config_dict: (DictConfig, dict),
                 func_dict: dict
                 ):
        super(MBMPC_TrainFlow, self).__init__(func_dict=func_dict)
        config = construct_dict_config(config_or_config_dict, obj=self)
        self.parameters = Parameters(source_config=config, parameters=dict())  # hyper parameter instance
        if train_sample_count_func:
            assert callable(train_sample_count_func)    # return TOTAL_AGENT_TRAIN_SAMPLE_COUNT

        from baconian.common.sampler.sample_data import MPC_TransitionData
        self.env = self.parameters('env')
        self.env_spec = self.env.env_spec
        env_spec = self.env_spec
        self.random_buffer = MPC_TransitionData(env_spec=env_spec,
                                                obs_shape=env_spec.obs_shape,
                                                action_shape=env_spec.action_shape,
                                                size=self.parameters('random_size'))
        self.rl_buffer = MPC_TransitionData(env_spec=env_spec,
                                                obs_shape=env_spec.obs_shape,
                                                action_shape=env_spec.action_shape,
                                                size=self.parameters('rl_size'))

    def _launch(self) -> bool:
        env = self.parameters('env')
        max_step = self.parameters('max_step')
        train_iter = self.parameters('train_iter')
        batch_size = self.parameters('batch_size')
        rand_rl_ratio = self.parameters('rand_rl_ratio')
        random_trajectory = self.parameters('random_trajectory')
        on_policy_trajectory = self.parameters('on_policy_trajectory')
        on_policy_iter = self.parameters('on_policy_iter')
        num_simulated_paths = self.parameters('num_simulated_paths')

        print("====> Preprocessing Data")   # normalization
        self.random_buffer, self.mean_dict, self.var_dict = self.data_preprocess(env, self.random_buffer, random_trajectory, max_step)

        print("====> Start Training")
        for iter in range(on_policy_iter):
            data = self.random_buffer.union(self.rl_buffer, rand_rl_ratio=rand_rl_ratio)
            batch_data_list = data.sample_batch_as_Transition(batch_size=batch_size, shuffle_flag=True, all_as_batch=True)
            for batch_data in batch_data_list:
                self._call_func('train', batch_data=batch_data) # dyna_epoch defined in agent
            self.rl_buffer = self._call_func('sample',
                                             env=env,
                                             sample_count=iter,
                                             buffer=self.rl_buffer,
                                             num_trajectory=on_policy_trajectory,
                                             max_step=max_step,
                                             num_simulated_paths=num_simulated_paths,
                                             in_which_status='TRAIN',
                                             store_flag=False)
        return True


    def random_buffer_sample(self, env, buffer, num_trajectory, max_step):
        '''RandomController.sample()'''
        for i in range(num_trajectory):
            obs = env.reset()
            ep_len = 0
            for j in range(max_step):
                act = self.env.action_space.sample()
                obs_, rew, done, _ = env.step(act)
                buffer.append(obs, act, obs_, done, rew)
                if done:
                    break
                else:
                    obs = obs_
        return buffer

    def data_preprocess(self, env, buffer, num_trajectory, max_step):
        buffer = self.random_buffer_sample(env, buffer, num_trajectory, max_step)
        normalized_buffer, mean_dict, var_dict = buffer.apply_normalization()
        return normalized_buffer, mean_dict, var_dict


from baconian.core.global_var import SinglentonStepCounter
from baconian.common.sampler.sample_data import TransitionData
import matplotlib.pyplot as plt

class DDPG_TrainTestFlow(Flow):
    '''
    For MBMF Test.
    '''
    required_func = ('train', 'test')
    required_key_dict = {
        "TRAIN_EVERY_SAMPLE_COUNT": 1,
        "TEST_EVERY_SAMPLE_COUNT": 1000,
        "START_TRAIN_AFTER_SAMPLE_COUNT": None,
        "START_TEST_AFTER_SAMPLE_COUNT": None,
    }

    def __init__(self,
                 train_sample_count_func,
                 config_or_config_dict: (DictConfig, dict),
                 func_dict: dict
                 ):
        super(DDPG_TrainTestFlow, self).__init__(func_dict=func_dict)
        config = construct_dict_config(config_or_config_dict, obj=self)
        self.parameters = Parameters(source_config=config, parameters=dict())
        if train_sample_count_func:
            assert callable(train_sample_count_func)

        self.env = self.parameters('env')
        self.env_spec = self.env.env_spec
        self.agent = self.parameters('agent')
        self.cyber = self.parameters('cyber')
        self.total_steps = self.parameters('total_steps')
        self.max_step_per_episode = self.parameters('max_step_per_episode')
        self.train_after_step = self.parameters('train_after_step')
        self.train_every_step = self.parameters('train_every_step')
        self.test_after_step = self.parameters('test_after_step')
        self.test_every_step = self.parameters('test_every_step')
        self.num_test = self.parameters('num_test')
        self.test_reward = []
        self.data_sample = []
        self.step_counter = SinglentonStepCounter(-1)

    def _launch(self) -> bool:
        env = self.env
        env_spec = self.env_spec
        cyber = self.cyber

        obs, ep_ret, ep_len = env.reset(), 0, 0
        for step in range(self.total_steps):
            self.step_counter.increase(1)
            act = self.agent.predict(obs=obs)
            obs_, reward, done, _ = cyber.step(obs, act)
            _buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, action_shape=env_spec.action_shape)
            _buffer.append(obs, act, obs_, done, reward)
            self.agent.algo.append_to_memory(_buffer)
            ep_ret += reward
            ep_len += 1

            if done or ep_len > self.max_step_per_episode:
                obs, ep_ret, ep_len = env.reset(), 0, 0
            else:
                obs = obs_

            if step > self.train_after_step and step % self.train_every_step == 0:
                self.agent.train()
            if step > self.test_after_step and step % self.test_every_step == 0:
                self.data_sample, self.test_reward = self.agent.test(env=env,
                                                                     cyber=cyber,
                                                                     data_sample=self.data_sample,
                                                                     test_reward=self.test_reward,
                                                                     num_test=self.num_test,
                                                                     max_step_per_episode=self.max_step_per_episode)
        env.close()
        self.plot_test_reward(self.data_sample, self.test_reward)
        return True

    @staticmethod
    def plot_test_reward(episode, reward):
        plt.plot(episode, reward)
        plt.ylabel('Reward')
        plt.xlabel('Data samples')
        plt.show()


def create_train_test_flow(train_every_sample_count=None,
                           test_every_sample_count=None,
                           start_train_after_sample_count=None,
                           start_test_after_sample_count=None,
                           train_func_and_args=None,
                           test_func_and_args=None,
                           sample_func_and_args=None,
                           train_samples_counter_func=None,
                           flow_type='TrainTestFlow',
                           **kwargs):
    # main parameters input
    config_dict = dict(
        TRAIN_EVERY_SAMPLE_COUNT=train_every_sample_count,
        TEST_EVERY_SAMPLE_COUNT=test_every_sample_count,
        START_TRAIN_AFTER_SAMPLE_COUNT=start_train_after_sample_count,
        START_TEST_AFTER_SAMPLE_COUNT=start_test_after_sample_count,
        **kwargs
    )

    def return_func_dict(s_dict):
        return dict(func=s_dict[0],
                    args=s_dict[1],
                    kwargs=s_dict[2])

    func_dict = dict(
        train=return_func_dict(train_func_and_args),
        test=return_func_dict(test_func_and_args),
        sample=return_func_dict(sample_func_and_args),
    )
    if train_samples_counter_func is None:
        def default_train_samples_counter_func():
            return get_global_status_collect()('TOTAL_AGENT_TRAIN_SAMPLE_COUNT')
        train_samples_counter_func = default_train_samples_counter_func

    if flow_type == 'TrainTestFlow':
        return TrainTestFlow(config_or_config_dict=config_dict,
                             train_sample_count_func=train_samples_counter_func, # default 500 samples in global config
                             func_dict=func_dict)
    elif flow_type == 'MBMPC_TrainFlow':
        return MBMPC_TrainFlow(config_or_config_dict=config_dict,
                               train_sample_count_func=train_samples_counter_func,
                               func_dict=func_dict)
    elif flow_type == 'DDPG_TrainTestFlow':
        return DDPG_TrainTestFlow(config_or_config_dict=config_dict,
                                  train_sample_count_func=train_samples_counter_func,
                                  func_dict=func_dict)
    else:
        raise ValueError("Please recheck flow_type.")