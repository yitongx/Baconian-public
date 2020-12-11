from baconian.common.data_pre_processing import RunningStandardScaler, IdenticalDataScaler, BatchStandardScaler, StandardScaler
from baconian.test.tests.set_up.setup import TestWithAll
import numpy as np
from baconian.envs.gym_env import make
from baconian.common.sampler.sample_data import TrajectoryData, TransitionData

class TestScaler(TestWithAll):
    '''
    Test Online update mean&var computation
    Compare with one-time computation
    '''
    def test_StandScaler(self):
        env = make('ModifiedHalfCheetah')
        env_spec = env.env_spec
        self.assertEqual(env_spec.flat_obs_dim, 18)
        self.assertEqual(env_spec.flat_action_dim, 6)

        buffer_size = 10
        buffer = TransitionData(env_spec=env_spec, obs_shape=env_spec.obs_shape, \
                       action_shape=env_spec.action_shape, size=buffer_size)
        obs = env.reset()
        for i in range(buffer_size):
            act = env.action_space.sample()
            obs_, rew, done, _ = env.step(act)
            buffer.append(obs, act, obs_, done, rew)

        batch_list = buffer.sample_batch_as_Transition(4, all_as_batch=True)
        state_input_scaler_1 = RunningStandardScaler(env_spec.flat_action_dim)

        for batch_data in batch_list:
            state_input_scaler_1.update_scaler(batch_data.action_set)

        mean_1 = state_input_scaler_1._mean
        var_1 = state_input_scaler_1._var

        print(mean_1)
        print(var_1)

        state_input_scaler_2 = RunningStandardScaler(env_spec.flat_action_dim)
        state_input_scaler_2.update_scaler(buffer.action_set)
        mean_2 = state_input_scaler_2._mean
        var_2 = state_input_scaler_2._var
        print(mean_2)
        print(var_2)

