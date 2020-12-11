from baconian.core.core import Basic
import abc
import numpy as np


class RewardFunc(Basic):
    allow_duplicate_name = True

    def __init__(self, name='reward_func'):
        super().__init__(name=name)

    @abc.abstractmethod
    def __call__(self, state, action, new_state, **kwargs) -> float:
        raise NotImplementedError

    def init(self):
        pass


class CostFunc(RewardFunc):
    pass


class RandomRewardFunc(RewardFunc):
    """
    Debug and test use only
    """

    def __init__(self, name='random_reward_func'):
        super().__init__(name)

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
        return np.random.random()


class MBMPC_HalfCheetah_CostFunc(RewardFunc):
    '''
    MBMPC cost func for Half Cheetah env
    '''
    def __init__(self, name='mb_mpc_halfcheetah_cost_func'):
        super().__init__(name)

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
        return self._cheetah_cost_fun(state=state, action=action, next_state=new_state)

    def _cheetah_cost_fun(self, state, action, next_state):
        if len(state.shape) > 1:
            heading_penalty_factor = 10
            scores = np.zeros((state.shape[0],))

            front_leg = state[:, 5]
            my_range = 0.2
            scores[front_leg >= my_range] += heading_penalty_factor

            front_shin = state[:, 6]
            my_range = 0
            scores[front_shin >= my_range] += heading_penalty_factor

            front_foot = state[:, 7]
            my_range = 0
            scores[front_foot >= my_range] += heading_penalty_factor

            scores -= (next_state[:, 17] - state[:, 17]) / 0.01 + 0.1 * (np.sum(action ** 2, axis=1))
            return scores

        heading_penalty_factor = 10
        score = 0

        # dont move front shin back so far that you tilt forward
        front_leg = state[5]
        my_range = 0.2
        if front_leg >= my_range:
            score += heading_penalty_factor

        front_shin = state[6]
        my_range = 0
        if front_shin >= my_range:
            score += heading_penalty_factor

        front_foot = state[7]
        my_range = 0
        if front_foot >= my_range:
            score += heading_penalty_factor

        score -= (next_state[17] - state[17]) / 0.01 + 0.1 * (np.sum(action ** 2))
        return score


class RewardFuncCostWrapper(CostFunc):
    '''
    Wrap reward func into cost func.
    '''

    def __init__(self, reward_func: RewardFunc, name='reward_func'):
        super().__init__(name)
        self._reward_func = reward_func

    def __call__(self, state, action, new_state, **kwargs) -> float:
        return self._reward_func.__call__(state=state,
                                          action=action,
                                          new_state=new_state) * -1.0

    def init(self):
        self._reward_func.init()


class QuadraticCostFunc(CostFunc):
    """
    A quadratic function
    """

    def __init__(self, C, c, name='reward_func'):
        """
        the cost is computed by 1/2[x_t, u_t].T *  C * [x_t, u_t] + [x_t, u_t] * c
        :param C: quadratic term
        :param c: linear term
        :param name:
        """
        super().__init__(name)
        self.C = np.array(C)
        self.c = np.array(c)
        self.state_action_flat_dim = self.C.shape[0]
        assert self.state_action_flat_dim == self.C.shape[1]
        assert len(self.C.shape) == 2
        assert self.c.shape[0] == self.state_action_flat_dim

    def __call__(self, state=None, action=None, new_state=None, **kwargs) -> float:
        u_s = np.concatenate((np.array(state).reshape(-1), np.array(action).reshape(-1))).reshape(
            self.state_action_flat_dim, 1)
        res = 0.5 * np.dot(np.dot(u_s.T, self.C), u_s) + np.dot(u_s.T, self.c).reshape(())
        return float(res)

if __name__ == '__main__':
    func = RandomRewardFunc(name='func')
    print(func())
