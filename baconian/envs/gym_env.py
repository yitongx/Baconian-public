from baconian.core.core import Env, EnvSpec
import gym.envs
from gym.envs.registration import registry
# do not remove the following import statements
import pybullet
import pybullet_envs

have_mujoco_flag = True
try:
    from gym.envs.mujoco import mujoco_env
except Exception:
    have_mujoco_flag = False
import numpy as np
import types
import gym.spaces as GymSpace
import baconian.common.spaces as garage_space
import gym.error as gym_error

_env_inited_count = dict()

# take multiple env into consideration
def make(gym_env_id: str, allow_multiple_env=True):
    """

    :param gym_env_id: gym environment id
    :type gym_env_id: int
    :param allow_multiple_env: allow multiple environments, by default True
    :type allow_multiple_env: bool
    :return: new gym environment
    :rtype: GymEnv
    """
    if gym_env_id == 'ModifiedHalfCheetah':
        return ModifiedHalfCheetahEnv(name=gym_env_id)

    if allow_multiple_env is True:
        if gym_env_id not in _env_inited_count:
            _env_inited_count[gym_env_id] = 0
        else:
            _env_inited_count[gym_env_id] += 1
        return GymEnv(gym_env_id, name='{}_{}'.format(gym_env_id, _env_inited_count[gym_env_id]))
    else:
        return GymEnv(gym_env_id)


def space_converter(space: GymSpace.Space):
    """
    Convert space into any one of "Box", "Discrete", or "Tuple" type.

    :param space: space of gym environment
    :type space: GymSpace
    :return: converted space
    :rtype: Box, Discrete, or Tuple
    """
    if isinstance(space, GymSpace.Box):
        return garage_space.Box(low=space.low, high=space.high)
    elif isinstance(space, GymSpace.Dict):
        return garage_space.Dict(space.spaces)
    elif isinstance(space, GymSpace.Discrete):
        return garage_space.Discrete(space.n)
    elif isinstance(space, GymSpace.Tuple):
        return garage_space.Tuple(list(map(space_converter, space.spaces)))
    else:
        raise NotImplementedError


class GymEnv(Env):
    """
    Gym environment wrapping module
    """
    _all_gym_env_id = list(registry.env_specs.keys())

    def __init__(self, gym_env_id: str, name: str = None):
        """

        :param gym_env_id: gym environment id
        :type gym_env_id: str
        :param name: name of the gym environment instance
        :type name: str
        """
        super().__init__(name=name if name else gym_env_id)
        self.env_id = gym_env_id
        try:
            self._gym_env = gym.make(gym_env_id)
        except gym_error.UnregisteredEnv:
            raise ValueError('Env id: {} is not supported currently'.format(gym_env_id))

        self._gym_env = gym.make(gym_env_id)
        self.action_space = space_converter(self._gym_env.action_space)
        self.observation_space = space_converter(self._gym_env.observation_space)

        if isinstance(self.action_space, garage_space.Box):
            self.action_space.low = np.nan_to_num(self.action_space.low)
            self.action_space.high = np.nan_to_num(self.action_space.high)
            self.action_space.sample = types.MethodType(self._sample_with_nan, self.action_space)
        if isinstance(self.observation_space, garage_space.Box):
            self.observation_space.low = np.nan_to_num(self.observation_space.low)
            self.observation_space.high = np.nan_to_num(self.observation_space.high)
            self.observation_space.sample = types.MethodType(self._sample_with_nan, self.observation_space)

        self.env_spec = EnvSpec(obs_space=self.observation_space, action_space=self.action_space)
        self.reward_range = self._gym_env.reward_range

    def step(self, action):
        """

        :param action: action to be taken by agent in the environment
        :type action: action to be taken by agent in the environment
        :return: step of the unwrapped environment
        :rtype: gym env
        """
        super().step(action)
        action = self.env_spec.flat_action(action)
        state, re, done, info = self.unwrapped.step(action=action)
        return state, re, bool(done), info

    def reset(self):
        """
        Reset the gym environment.
        :return:
        """
        super().reset()

        return self.unwrapped.reset()

    def init(self):
        """
        Initialize the gym environment.
        :return:
        """
        super().init()

        return self.reset()

    def seed(self, seed=None):
        """

        :param seed: seed of random number generalization
        :type seed: int
        :return: seed of the unwrapped environment
        :rtype: int
        """
        return super().seed(seed)

    def get_state(self):
        """

        :return:the state of unwrapped gym environment
        :rtype: np.ndarray
        """
        if (have_mujoco_flag is True and isinstance(self.unwrapped_gym, mujoco_env.MujocoEnv)) \
                or (hasattr(self.unwrapped_gym, '_get_obs') and callable(self.unwrapped_gym._get_obs)):
            return self.unwrapped_gym._get_obs()
        elif hasattr(self.unwrapped_gym, '_get_ob') and callable(self.unwrapped_gym._get_ob):
            return self.unwrapped_gym._get_ob()
        elif hasattr(self.unwrapped_gym, 'state'):
            return self.unwrapped_gym.state if isinstance(self.unwrapped_gym.state, np.ndarray) else np.array(
                self.unwrapped_gym.state)
        elif hasattr(self.unwrapped_gym, 'observation'):
            return self.unwrapped_gym.observation if isinstance(self.unwrapped_gym.observation, np.ndarray) \
                else np.array(self.unwrapped_gym.state)
        elif hasattr(self.unwrapped_gym, 'spec') and hasattr(self.unwrapped_gym.spec, 'id') and self.unwrapped_gym.spec.id in specialEnv:
            return specialEnv[self.unwrapped_gym.spec.id](self)
        elif hasattr(self.unwrapped_gym, 'robot'):
            return self.unwrapped_gym.robot.calc_state()
        else:
            raise ValueError('Env id: {} is not supported for method get_state'.format(self.env_id))

    @property
    def unwrapped(self):
        """
        :return: original unwrapped gym environment
        :rtype: gym env
        """
        return self._gym_env

    @property
    def unwrapped_gym(self):
        """

        :return: gym environment, depend on attribute 'unwrapped'
        :rtype: gym env
        """
        if hasattr(self._gym_env, 'unwrapped'):
            return self._gym_env.unwrapped
        else:
            return self._gym_env

    @staticmethod
    def _sample_with_nan(space: garage_space.Space):
        """

        :param space: a 'Box'type space
        :return: numpy clip of space that contains nan values
        :rtype: np.ndarray
        """
        assert isinstance(space, garage_space.Box)
        high = np.ones_like(space.low)
        low = -1 * np.ones_like(space.high)
        return np.clip(np.random.uniform(low=low, high=high, size=space.low.shape),
                       a_min=space.low,
                       a_max=space.high)

    def __str__(self):
        return "<GymEnv instance> {}".format(self.env_id)


def get_lunarlander_state(env):
    pos = env.unwrapped_gym.lander.position
    vel = env.unwrapped_gym.lander.linearVelocity
    fps = 50
    scale = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
    leg_down = 18
    viewport_w = 600
    viewport_h = 400
    state = [
        (pos.x - viewport_w / scale / 2) / (viewport_w / scale / 2),
        (pos.y - (env.unwrapped_gym.helipad_y + leg_down / scale)) / (viewport_h / scale / 2),
        vel.x * (viewport_w / scale / 2) / fps,
        vel.y * (viewport_h / scale / 2) / fps,
        env.unwrapped_gym.lander.angle,
        20.0 * env.unwrapped_gym.lander.angularVelocity / fps,
        1.0 if env.unwrapped_gym.legs[0].ground_contact else 0.0,
        1.0 if env.unwrapped_gym.legs[1].ground_contact else 0.0
    ]
    return np.array(state, dtype=np.float32)


specialEnv = {
    'LunarLander-v2': get_lunarlander_state
}


'''
Modified Half Cheetah with Obseravtion(18,) in mujoco_env
'''
import os
from gym import utils
from baconian.core.status import *
from baconian.core.core import Basic
from baconian.common.spaces import Space
from baconian.common.logging import Recorder
from baconian.common.special import flat_dim, flatten
from baconian.config.global_config import GlobalConfig
from baconian.core.util import register_name_globally, init_func_arg_record_decorator


class ModifiedHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle, Env):
    key_list = ()
    required_key_dict = ()
    STATUS_LIST = ('JUST_RESET', 'INITED', 'TRAIN', 'TEST', 'CREATED')
    INIT_STATUS = 'CREATED'
    allow_duplicate_name = False

    def __init__(self, name='ModifiedHalfCheetah'):
        print('====> Initiating Modified Half Cheetah with observation space Box(18,)')

        Basic.__init__(self, name=name, status=StatusWithSubInfo(obj=self))

        self.action_space = None
        self.observation_space = None
        self._last_reset_point = 0
        self.trajectory_level_step_count = 0
        self.recorder = Recorder(default_obj=self)
        self.total_step_count_fn = lambda: \
            self._status.group_specific_info_key(info_key='step', group_way='sum')  # record step
        self.env_spec = None
        self._inited_flag = False   # avoid multiple inits

        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/modified_half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

        self.init()

    @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='INITED')
    def init(self):
        if self._inited_flag:
            print('Warning: Current env has been initialized. Check if env.inited() has been called multiple times')
            print('Warning: Duplicated env initialization has been ignored')
            return
        self._status.set_status('INITED')


        self.action_space = space_converter(self.action_space)
        self.observation_space = space_converter(self.observation_space)
        if isinstance(self.action_space, garage_space.Box):
            self.action_space.low = np.nan_to_num(self.action_space.low)
            self.action_space.high = np.nan_to_num(self.action_space.high)
            self.action_space.sample = types.MethodType(self._sample_with_nan, self.action_space)
        if isinstance(self.observation_space, garage_space.Box):
            self.observation_space.low = np.nan_to_num(self.observation_space.low)
            self.observation_space.high = np.nan_to_num(self.observation_space.high)
            self.observation_space.sample = types.MethodType(self._sample_with_nan, self.observation_space)

        self.env_spec = EnvSpec(obs_space=self.observation_space, action_space=self.action_space)
        self._inited_flag = True

    @register_counter_info_to_status_decorator(increment=1, info_key='step', under_status=('TRAIN', 'TEST'),
                                               ignore_wrong_status=True)
    def step(self, action):
        # TODO: 9.5 is there need to flat_action?
        # action = self.env_spec.flat_action(action)
        self.trajectory_level_step_count += 1
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        reward_run = ob[0] - 0.0 * np.square(ob[2])
        reward_ctrl = -0.1 * np.square(action).sum()    # -0.05 for RL in Half Cheetah
        reward = reward_run + reward_ctrl
        done = False # No specific terminal state

        return ob, reward, done, {}

    @register_counter_info_to_status_decorator(increment=1, info_key='reset', under_status='JUST_RESET')
    def reset(self):
        self._status.set_status('JUST_RESET')
        self._last_reset_point = self.total_step_count_fn()
        self.trajectory_level_step_count = 0
        self._status.reset()    # TODO: 9.4 test whether there is need to reset self._status
        self._inited_flag = False

        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def get_state(self):
        return self._get_obs()

    def reset_model(self):  # inherited from mujoco.env
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

    def seed(self, seed=None):
        super().seed(seed)

    @property
    def unwrapped(self):
        return self

    @staticmethod
    def _sample_with_nan(space: garage_space.Space):
        """

        :param space: a 'Box'type space
        :return: numpy clip of space that contains nan values
        :rtype: np.ndarray
        """
        assert isinstance(space, garage_space.Box)
        high = np.ones_like(space.low)
        low = -1 * np.ones_like(space.high)
        return np.clip(np.random.uniform(low=low, high=high, size=space.low.shape),
                       a_min=space.low,
                       a_max=space.high)

    def __str__(self):
        return '<ModifiedHalfCheetahEnv instance> {}'.format(self.name)