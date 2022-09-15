import math

import simple_ml as sm


def uniform(x, low=0.0, high=1.0):

    x.data = sm.randu(x.shape, low=low, high=high, device=x.device)


def normal(x, mean=0.0, std=1.0):

    x.data = sm.randn(x.shape, mean=mean, std=std, device=x.device)


def constant(x, c=0.0):

    x.data = sm.full(shape=x.shape, fill_value=c, device=x.device)


def ones(x):

    x.data = sm.ones(shape=x.shape, device=x.device)


def zeros(x):

    x.data = sm.zeros(shape=x.shape, device=x.device)


def _calculate_fans(x):

    # Note: this function assumes that if the shape of x is four dimensional
    #       then we are trying to compute the fan size for a convolution layer's
    #       weights
    fan_in, fan_out = x.shape[-2:]
    if len(x.shape) == 4:
        # x is of shape (k, k, c_in, c_out)
        k2 = x.shape[0] * x.shape[1]
        fan_in *= float(k2)
        fan_out *= float(k2)
    return fan_in, fan_out


def xavier_uniform(x, gain=1.0):

    fan_in, fan_out = _calculate_fans(x)
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    uniform(x, low=-a, high=a)


def xavier_normal(x, gain=1.0):

    fan_in, fan_out = _calculate_fans(x)
    a = gain * math.sqrt(2 / (fan_in + fan_out))
    normal(x, std=a)


def kaiming_uniform(x, mode="fan_in", nonlinearity="relu"):

    fan_in, fan_out = _calculate_fans(x)
    fan_mode = fan_in if mode == "fan_in" else fan_out
    # only nonlinearity specified is ReLU
    # update this if we get other nonlinearities
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_mode)
    uniform(x, low=-bound, high=bound)


def kaiming_normal(x, mode="fan_in", nonlinearity="relu"):

    fan_in, fan_out = _calculate_fans(x)
    fan_mode = fan_in if mode == "fan_in" else fan_out
    # only nonlinearity specified is ReLU
    # update this if we get other nonlinearities
    gain = math.sqrt(2)
    bound = gain / math.sqrt(fan_mode)
    normal(x, std=bound)
