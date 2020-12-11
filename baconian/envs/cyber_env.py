import os
import collections
import numpy as np
import tensorflow as tf
from enum import Enum
from baconian.core.core import EnvSpec, Env


class EnvName(Enum):
    PendulumEnv = 'PendulumEnv'
    HalfCheetahEnv = 'HalfCheetahEnv'
    ModifiedHalfCheetahEnv = 'ModifiedHalfCheetahEnv'


class CyberEnv(Env):
    '''
    Load model from model_path & wrap as a baconian Env
    '''
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    def __init__(self,
                 env: Env,
                 env_name: EnvName,
                 epoch_to_use: int,
                 use_traj_input: bool,
                 use_mbmf: bool,
                 model_path=None):

        super().__init__(name='env_name')
        self.use_mbmf = use_mbmf
        if self.use_mbmf:
            model_folder = 'mbmf_mlp_dynamics_model'
        else:
            model_folder = 'mlp_dynamics_model'

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.env_spec = env.env_spec

        ### Load trained MPC model
        assert model_path
        MODEL_PATH = model_path
        saver = tf.train.import_meta_graph(MODEL_PATH + f'/{model_folder}_{env_name.value}/-{epoch_to_use}.meta')
        saver.restore(self.sess, MODEL_PATH + f'/{model_folder}_{env_name.value}/-{epoch_to_use}')

        self.output_ = tf.get_collection('delta_obs')[0]
        graph = tf.get_default_graph()
        self.input_ = graph.get_operation_by_name('input').outputs[0]
        self.use_traj_input = use_traj_input
        self.past_input_mem = None

        if self.use_traj_input:
            self.init_memory()

    def init_memory(self):
        self.past_input_mem = collections.deque(
            [np.zeros((1, self.input_.shape[2]), dtype=np.float64) for _ in range(self.input_.shape[1])],
            int(self.input_.shape[1])
        )

    def process_input(self, obs, act):
        if obs.ndim < act.ndim:
            act = np.squeeze(act, axis=0)
        assert obs.ndim == act.ndim
        input_ = np.concatenate((obs, act), axis=0).reshape(1, -1)
        if self.use_traj_input:
            self.past_input_mem.append(input_)
            return np.expand_dims(np.concatenate(list(self.past_input_mem), axis=0), axis=0)
        else:
            return input_

    def step(self, obs, act):
        super().step(act)
        pass
        # raise NotImplementedError

    def calculate_reward(self, obs, act, obs_):
        raise NotImplementedError

    def get_predicted_new_state(self, input_, obs):
        '''
        Internal usage.
        :param input_:
        :param obs:
        :return:
        '''
        if self.use_mbmf:
            return self.sess.run(self.output_, feed_dict={self.input_: input_})[0]
        else:
            return self.sess.run(self.output_, feed_dict={self.input_: input_})[0]


class PendulumnCyber(CyberEnv):
    def __init__(self, env, epoch_to_use=60, use_traj_input=False, use_mbmf=False, model_path=None):
        super().__init__(env, EnvName.PendulumEnv, epoch_to_use, use_traj_input, use_mbmf, model_path=model_path)

    def step(self, obs, act):
        super().step(obs, act)
        cost = self.calculate_reward(obs, act, None)
        input_ = self.process_input(obs, act)
        obs_ = self.get_predicted_new_state(input_, obs)
        return obs_, -cost, False, {}

    @classmethod
    def calculate_reward(cls, obs, act, obs_):
        th, thdot = np.arctan(obs[1] / obs[0]), obs[2]
        return cls.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (act[0] ** 2)

    @classmethod
    def angle_normalize(cls, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi


'''class HalfCheetahCyber(CyberEnv):
    def __init__(self, epoch_to_use=50, use_traj_input=False, use_mbmf=False):
        super().__init__(EnvName.HalfCheetahEnv, epoch_to_use, use_traj_input, use_mbmf)

    def step(self, obs, act):
        input_ = self.process_input(obs, act)
        obs_ = self.get_predicted_new_state(input_, obs)
        reward = self.calculate_reward(obs, act, obs_)
        done = False
        return obs_, reward, done, {}

    @classmethod
    def calculate_reward(cls, obs, act, obs_):
        reward_ctrl = -0.1 * np.square(act).sum()
        reward_run = obs_[0] - 0.0 * np.square(obs_[2])
        return reward_run + reward_ctrl'''