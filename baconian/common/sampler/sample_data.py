from baconian.common.special import *
from baconian.core.core import EnvSpec
from copy import deepcopy
import typeguard as tg
from baconian.common.error import *


class SampleData(object):
    def __init__(self,
                 env_spec: EnvSpec = None,
                 obs_shape=None,
                 action_shape=None):
        if env_spec is None and (obs_shape is None or action_shape is None):
            raise ValueError('At least env_spec or (obs_shape, action_shape) should be passed in')
        self.env_spec = env_spec
        self.obs_shape = env_spec.obs_shape if env_spec else obs_shape
        self.action_shape = env_spec.action_shape if env_spec else action_shape

    def reset(self):
        raise NotImplementedError

    def append(self, *args, **kwargs):
        raise NotImplementedError

    def union(self, sample_data):
        raise NotImplementedError

    def get_copy(self):
        raise NotImplementedError

    def __call__(self, set_name, **kwargs):
        raise NotImplementedError

    def append_new_set(self, name, data_set: (list, np.ndarray), shape: (tuple, list)):
        raise NotImplementedError

    def sample_batch(self, *args, **kwargs):
        raise NotImplementedError

    def sample_batch_as_Transition(self, *args, **kwargs):
        raise NotImplementedError

    def apply_transformation(self, set_name, func, *args, **kwargs):
        raise NotImplementedError

    def apply_op(self, set_name, func, *args, **kwargs):
        raise NotImplementedError


class TransitionData(SampleData):
    def __init__(self,
                 env_spec: EnvSpec = None,
                 obs_shape=None,
                 action_shape=None,
                 size=None):
        super(TransitionData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)

        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0
        assert isinstance(self.obs_shape, (list, tuple))
        assert isinstance(self.action_shape, (list, tuple))
        self.obs_shape = list(self.obs_shape)
        self.action_shape = list(self.action_shape)

        self._internal_data_dict = {
            'state_set': [np.empty([0] + self.obs_shape), self.obs_shape],  # [(0, obs_shape), [obs_shape]]
            'new_state_set': [np.empty([0] + self.obs_shape), self.obs_shape],
            'action_set': [np.empty([0] + self.action_shape), self.action_shape],
            'reward_set': [np.empty([0]), []],
            'done_set': [np.empty([0], dtype=bool), []]
        }
        self.current_index = 0
        self.current_size = 0
        self.max_size = size if size else None

    def __len__(self):
        return len(self._internal_data_dict['state_set'][0])

    def __call__(self, set_name, **kwargs):
        if set_name not in self._allowed_data_set_keys:
            raise ValueError('pass in set_name within {} '.format(self._allowed_data_set_keys))
        return make_batch(self._internal_data_dict[set_name][0],
                          original_shape=self._internal_data_dict[set_name][1])

    def reset(self):
        for key, data_set in self._internal_data_dict.items():
            self._internal_data_dict[key][0] = np.empty([0, *self._internal_data_dict[key][1]])
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self,
               state: np.ndarray,
               action: np.ndarray,
               new_state: np.ndarray,
               done: bool,
               reward: float):
        if self.max_size:
            self.cumulative_reward += reward
            if self.current_size == self.max_size:
                self._internal_data_dict['state_set'][0][self.current_index] = np.reshape(state, [1] + self.obs_shape)
                self._internal_data_dict['new_state_set'][0][self.current_index] = np.reshape(new_state, [1] + self.obs_shape)
                self.cumulative_reward -= self._internal_data_dict['reward_set'][0][self.current_index]
                self._internal_data_dict['reward_set'][0][self.current_index] = np.reshape(reward, [1])
                self._internal_data_dict['done_set'][0][self.current_index] = np.reshape(np.array(done, dtype=bool), [1])
                self._internal_data_dict['action_set'][0][self.current_index] = np.reshape(action, [1] + self.action_shape)
            else:
                self._internal_data_dict['state_set'][0] = np.concatenate(
                    (self._internal_data_dict['state_set'][0], np.reshape(state, [1] + self.obs_shape)), axis=0)
                self._internal_data_dict['new_state_set'][0] = np.concatenate(
                    (self._internal_data_dict['new_state_set'][0], np.reshape(new_state, [1] + self.obs_shape)), axis=0)
                self._internal_data_dict['reward_set'][0] = np.concatenate(
                    (self._internal_data_dict['reward_set'][0], np.reshape(reward, [1])), axis=0)
                self._internal_data_dict['done_set'][0] = np.concatenate(
                    (self._internal_data_dict['done_set'][0], np.reshape(np.array(done, dtype=bool), [1])), axis=0)
                self._internal_data_dict['action_set'][0] = np.concatenate(
                    (self._internal_data_dict['action_set'][0], np.reshape(action, [1] + self.action_shape)), axis=0)
            self.current_index = (self.current_index + 1) % self.max_size
            self.current_size = min(self.current_size + 1, self.max_size)

        else:
            self._internal_data_dict['state_set'][0] = np.concatenate(
                (self._internal_data_dict['state_set'][0], np.reshape(state, [1] + self.obs_shape)), axis=0)
            self._internal_data_dict['new_state_set'][0] = np.concatenate(
                (self._internal_data_dict['new_state_set'][0], np.reshape(new_state, [1] + self.obs_shape)), axis=0)
            self._internal_data_dict['reward_set'][0] = np.concatenate(
                (self._internal_data_dict['reward_set'][0], np.reshape(reward, [1])), axis=0)
            self._internal_data_dict['done_set'][0] = np.concatenate(
                (self._internal_data_dict['done_set'][0], np.reshape(np.array(done, dtype=bool), [1])), axis=0)
            self._internal_data_dict['action_set'][0] = np.concatenate(
                (self._internal_data_dict['action_set'][0], np.reshape(action, [1] + self.action_shape)), axis=0)
            self.cumulative_reward += reward

    def union(self, sample_data):
        # TODO: 9.18 auto scale the original buffer when not every buffer has limited size.
        #  auto_scale_flag
        assert isinstance(sample_data, type(self))
        self.cumulative_reward += sample_data.cumulative_reward
        self.step_count_per_episode += sample_data.step_count_per_episode
        for key, val in self._internal_data_dict.items():
            assert self._internal_data_dict[key][1] == sample_data._internal_data_dict[key][1]
            self._internal_data_dict[key][0] = np.concatenate(
                (self._internal_data_dict[key][0], sample_data._internal_data_dict[key][0]), axis=0)

    def get_copy(self):
        obj = type(self)(env_spec=self.env_spec, obs_shape=self.obs_shape, action_shape=self.action_shape, size=self.max_size)
        for key in self._internal_data_dict:
            obj._internal_data_dict[key] = deepcopy(self._internal_data_dict[key])
        return obj

    def append_new_set(self, name, data_set: (list, np.ndarray), shape: (tuple, list)):
        assert len(data_set) == len(self)
        assert len(np.array(data_set).shape) - 1 == len(shape)
        if len(shape) > 0:
            assert np.equal(np.array(data_set).shape[1:], shape).all()
        shape = tuple(shape)
        self._internal_data_dict[name] = [np.array(data_set), shape]


    def sample_batch(self, batch_size, shuffle_flag=True, **kwargs) -> dict:
        '''
        Sample one batch as dictionary.

        :param batch_size:
        :param shuffle_flag:
        :param kwargs:
        :return:
        '''
        total_num = len(self)
        # id_index = np.random.randint(low=0, high=total_num, size=batch_size)
        assert batch_size <= total_num
        id_index = np.random.permutation(total_num)[:batch_size] if shuffle_flag else np.arange(batch_size)

        batch_data = dict()
        for key in self._internal_data_dict.keys():
            batch_data[key] = self(key)[id_index]
        return batch_data

    def sample_batch_as_Transition(self, batch_size, shuffle_flag=True, all_as_batch=True):
        '''
        Two types of usage. Switch through 'all_as_batch'
        1. Sample batch_size data from shuffled Transition and may sample repeated data, which is not commonly used.
        2. Sample whole data as batch data with/without random shuffle within each batch, which covers entire data and
        is more commonly used. return as list compose of Transition.

        :param batch_size:
        :param shuffle_flag:
        :param all_as_batch:
        :return:
        '''
        if not shuffle_flag:
            raise NotImplementedError

        total_num = len(self)
        if not all_as_batch:
            # 9.9 what if batch_size > total_num?
            id_index = np.random.randint(low=0, high=total_num, size=batch_size)
            batch_data = type(self)(env_spec=self.env_spec, obs_shape=self.obs_shape, action_shape=self.action_shape)
            for key in self._internal_data_dict.keys():
                batch_data._internal_data_dict[key][0] = self._internal_data_dict[key][0][id_index]
            assert len(batch_data) == batch_size
            assert (type(batch_data) == type(self))
            return batch_data
        else:
            batch_data_list = []
            id_index = np.random.permutation(total_num)
            for key in self._internal_data_dict.keys():
                self._internal_data_dict[key][0] = self._internal_data_dict[key][0][id_index]
            for i in range(0, total_num, batch_size):
                real_size = batch_size if i + batch_size < total_num else total_num - i * batch_size
                batch_data = type(self)(env_spec=self.env_spec, obs_shape=self.obs_shape,
                                            action_shape=self.action_shape, size=real_size)
                for key in self._internal_data_dict.keys():
                    batch_data._internal_data_dict[key][0] = self._internal_data_dict[key][0][i:i+batch_size]
                batch_data_list.append(batch_data)
            return batch_data_list

    def get_mean_of(self, set_name):
        return self.apply_op(set_name=set_name, func=np.mean)

    def get_sum_of(self, set_name):
        return self.apply_op(set_name=set_name, func=np.sum)

    def apply_transformation(self, set_name, func, direct_apply=False, **func_kwargs):
        # TODO: 9.18 (later): Add multi-input & output, return as dict.
        data = make_batch(self._internal_data_dict[set_name][0],
                          original_shape=self._internal_data_dict[set_name][1])
        transformed_data = make_batch(func(data, **func_kwargs),
                                      original_shape=self._internal_data_dict[set_name][1])
        if transformed_data.shape != data.shape:
            raise TransformationResultedToDifferentShapeError()
        elif direct_apply is True:
            self._internal_data_dict[set_name][0] = transformed_data
        return transformed_data

    def apply_op(self, set_name, func, **func_kwargs):
        data = make_batch(self._internal_data_dict[set_name][0],
                          original_shape=self._internal_data_dict[set_name][1])
        applied_op_data = np.array(func(data, **func_kwargs))
        return applied_op_data

    def shuffle(self, index: list = None):
        if not index:
            index = np.arange(len(self._internal_data_dict['state_set'][0]))
            np.random.shuffle(index)
        for key in self._internal_data_dict.keys():
            self._internal_data_dict[key][0] = self._internal_data_dict[key][0][index]

    def apply_normalization(self, set_name=None):
        '''
        Apply normalization to assigned sets in Transition
        return transformed Transition and mean-var list of original sets
        '''
        mean_dict = dict()
        var_dict = dict()
        if not set_name:
            keys = ['state_set', 'new_state_set', 'action_set']
        else:
            keys = set_name
            assert isinstance(keys, list)


        for key in keys:
            assert key in self._internal_data_dict.keys()

        transformed_buffer = self.get_copy()
        for key in keys:
            _mean = np.mean(transformed_buffer._internal_data_dict[key][0])
            _var = np.var(transformed_buffer._internal_data_dict[key][0])
            transformed_buffer._internal_data_dict[key][0] = (transformed_buffer._internal_data_dict[key][0] - _mean) / (_var + 1e-10)
            mean_dict[key] = _mean
            var_dict[key] = _var

        return transformed_buffer, mean_dict, var_dict

    def apply_denormalization(self, set_name, mean_dict, var_dict):
        '''
        Apply denormalization to assigned sets in Transition
        return transformed Transition
        '''

        if not set_name:
            keys = ['state_set', 'new_state_set', 'action_set']
        else:
            keys = set_name
            assert isinstance(keys, list)

        for key in keys:
            assert key in self._internal_data_dict.keys() and key in mean_dict and key in var_dict

        transformed_buffer = self.get_copy()
        for key in keys:
            _mean = mean_dict[key]
            _var = var_dict[key]
            transformed_buffer._internal_data_dict[key][0] = transformed_buffer._internal_data_dict[key][0] * (_var + 1e-10) + _mean

        return transformed_buffer

    @property
    def _allowed_data_set_keys(self):
        return list(self._internal_data_dict.keys())

    @property
    def state_set(self):
        return self('state_set')

    @property
    def new_state_set(self):
        return self('new_state_set')

    @property
    def action_set(self):
        return self('action_set')

    @property
    def reward_set(self):
        return self('reward_set')

    @property
    def done_set(self):
        return self('done_set')


class MPC_TransitionData(TransitionData):
    '''
    TransitionData with dagger aggregation used in MBMPC.
    '''
    def __init__(self,
                 env_spec: EnvSpec = None,
                 obs_shape=None,
                 action_shape=None,
                 size=None):
        super(MPC_TransitionData, self).__init__(env_spec=env_spec,
                                                 obs_shape=obs_shape,
                                                 action_shape=action_shape,
                                                 size=size)
    def union(self, d_rl, rand_rl_ratio: float = 0.1):
        '''
        Use dagger to union first buffer and second buffer
        Default 'rand_rl_ratio' refers to first_buffer: second_buffer
        Return TransitionData for training
        '''
        assert isinstance(d_rl, type(self))

        train_dict = TransitionData(env_spec=self.env_spec, obs_shape=self.obs_shape, action_shape=self.action_shape)
        d_rand = self
        rl_data_dict = d_rl
        rl_data_len = len(rl_data_dict)

        if rl_data_len == 0:
            train_dict = d_rand
        else:
            rand_data_len = int(rl_data_len * rand_rl_ratio / (1 - rand_rl_ratio))
            rand_data_dict = self.sample_batch_as_Transition(batch_size=rand_data_len, shuffle_flag=True, all_as_batch= False)
            shuffler = np.random.permutation(rand_data_len + rl_data_len)
            for key in self._internal_data_dict.keys():
                train_dict._internal_data_dict[key][0] = np.concatenate((rand_data_dict._internal_data_dict[key][0], \
                                                                         rl_data_dict._internal_data_dict[key][0]), axis=0)
                train_dict._internal_data_dict[key][0] = train_dict._internal_data_dict[key][0][shuffler]

        return train_dict


class TrajectoryData(SampleData):
    def __init__(self, env_spec=None, obs_shape=None, action_shape=None):
        super(TrajectoryData, self).__init__(env_spec=env_spec, obs_shape=obs_shape, action_shape=action_shape)
        self.trajectories = []

    def reset(self):
        self.trajectories = []

    def append(self, transition_data: TransitionData):
        self.trajectories.append(transition_data)

    def union(self, sample_data):
        if not isinstance(sample_data, type(self)):
            raise TypeError()
        self.trajectories += sample_data.trajectories

    def return_as_transition_data(self, shuffle_flag=False) -> TransitionData:
        transition_set = self.trajectories[0].get_copy()
        for i in range(1, len(self.trajectories)):
            transition_set.union(self.trajectories[i])
        if shuffle_flag is True:
            transition_set.shuffle()
        return transition_set

    def get_mean_of(self, set_name):
        tran = self.return_as_transition_data()
        return tran.get_mean_of(set_name)

    def get_sum_of(self, set_name):
        tran = self.return_as_transition_data()
        return tran.get_sum_of(set_name)

    def __len__(self):
        return len(self.trajectories)

    def get_copy(self):
        tmp_traj = TrajectoryData(env_spec=self.env_spec, obs_shape=self.obs_shape, action_shape=self.action_shape)
        for traj in self.trajectories:
            tmp_traj.append(transition_data=traj.get_copy())
        return tmp_traj

    def apply_transformation(self, set_name, func, direct_apply=False, **func_kwargs):
        # TODO unit test
        for traj in self.trajectories:
            traj.apply_transformation(set_name, func, direct_apply, **func_kwargs)

    def apply_op(self, set_name, func, **func_kwargs):
        # TODO unit test
        res = []
        for traj in self.trajectories:
            res.append(traj.apply_op(set_name, func, **func_kwargs))
        return np.array(res)
