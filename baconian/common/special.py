"""
This script is from garage
"""
import gym.spaces
import numpy as np
import scipy
import scipy.signal
from typeguard import typechecked
import baconian.common.spaces as mbrl_spaces


def weighted_sample(weights, objects):
    """
    Return a random item from objects, with the weighting defined by weights
    (which must sum to 1).
    """
    # An array of the weights, cumulatively summed.
    cs = np.cumsum(weights)
    # Find the index of the first weight over a random value.
    idx = sum(cs < np.random.rand())
    return objects[min(idx, len(objects) - 1)]


def weighted_sample_n(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r.reshape((-1, 1))).sum(axis=1)
    n_items = len(items)
    return items[np.minimum(k, n_items - 1)]


# compute softmax for each row
def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    expx = np.exp(shifted)
    return expx / np.sum(expx, axis=-1, keepdims=True)


# compute entropy for each row
def cat_entropy(x):
    return -np.sum(x * np.log(x), axis=-1)


# compute perplexity for each row
def cat_perplexity(x):
    return np.exp(cat_entropy(x))


def explained_variance_1d(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1
    return ret


def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret


def from_onehot(v):
    return np.nonzero(v)[0][0]


def from_onehot_n(v):
    if ((isinstance(v, np.ndarray) and not v.size)
            or (isinstance(v, list) and not v)):
        return []
    return np.nonzero(v)[1]


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
    return np.sum(x * (discount ** np.arange(len(x))))


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


def make_batch(v, original_shape: (list, tuple)):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    # assert len(v.shape) <= len(original_shape) + 1
    if len(v.shape) == len(original_shape) + 1 and np.equal(np.array(v.shape[1:]),
                                                            np.array(original_shape)).all() is True:
        return v
    else:
        bs = np.prod(list(v.shape)) / np.prod(original_shape)
        assert float(bs).is_integer()
        return np.reshape(v, newshape=[int(bs)] + list(original_shape))


def flat_dim(space):
    if isinstance(space, mbrl_spaces.Box):
        return np.prod(space.low.shape)
    elif isinstance(space, mbrl_spaces.Discrete):
        return space.n
    elif isinstance(space, mbrl_spaces.Tuple):
        return np.sum([flat_dim(x) for x in space.spaces])
    else:
        raise NotImplementedError

# flatten(action_space, action)
def flatten(space, obs, one_hot_for_discrete=False):
    if isinstance(space, mbrl_spaces.Box):
        return np.asarray(obs).flatten()
    elif isinstance(space, mbrl_spaces.Discrete):
        if one_hot_for_discrete is True:
            if space.n == 2:
                obs = int(obs)
            return to_onehot(obs, space.n)
        else:
            return int(obs)
    elif isinstance(space, mbrl_spaces.Tuple):
        return np.concatenate(
            [flatten(c, xi) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


def flatten_n(space, obs):
    if isinstance(space, mbrl_spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], -1))
    elif isinstance(space, mbrl_spaces.Discrete):
        return to_onehot_n(np.array(obs, dtype=np.int), space.n)
    elif isinstance(space, mbrl_spaces.Tuple):
        obs_regrouped = [[obs[i] for o in obs] for i in range(len(obs[0]))]
        flat_regrouped = [
            flatten_n(c, oi) for c, oi in zip(space.spaces, obs_regrouped)
        ]
        return np.concatenate(flat_regrouped, axis=-1)
    else:
        raise NotImplementedError


def unflatten(space, obs):
    if isinstance(space, mbrl_spaces.Box):
        return np.asarray(obs).reshape(space.shape)
    elif isinstance(space, mbrl_spaces.Discrete):
        return from_onehot(np.array(obs, dtype=np.int))
    elif isinstance(space, mbrl_spaces.Tuple):
        dims = [flat_dim(c) for c in space.spaces]
        flat_xs = np.split(obs, np.cumsum(dims)[:-1])
        return tuple(unflatten(c, xi) for c, xi in zip(space.spaces, flat_xs))
    else:
        raise NotImplementedError


def unflatten_n(space, obs):
    if isinstance(space, mbrl_spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0],) + space.shape)
    elif isinstance(space, mbrl_spaces.Discrete):
        return from_onehot_n(np.array(obs, dtype=np.int))
    elif isinstance(space, mbrl_spaces.Tuple):
        dims = [flat_dim(c) for c in space.spaces]
        flat_xs = np.split(obs, np.cumsum(dims)[:-1], axis=-1)
        unflat_xs = [
            unflatten_n(c, xi) for c, xi in zip(space.spaces, flat_xs)
        ]
        unflat_xs_grouped = list(zip(*unflat_xs))
        return unflat_xs_grouped
    else:
        raise NotImplementedError
