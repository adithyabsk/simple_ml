import math

import needle as ndl


def uniform(x, low=0.0, high=1.0):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.randu(x.shape, low=low, high=high, device=x.device)
    ### END YOUR SOLUTION


def normal(x, mean=0.0, std=1.0):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.randn(x.shape, mean=mean, std=std, device=x.device)
    ### END YOUR SOLUTION


def constant(x, c=0.0):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.full(shape=x.shape, fill_value=c, device=x.device)
    ### END YOUR SOLUTION


def ones(x):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.ones(shape=x.shape, device=x.device)
    ### END YOUR SOLUTION


def zeros(x):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.zeros(shape=x.shape, device=x.device)
    ### END YOUR SOLUTION


def _calculate_fans(x):
    ### BEGIN YOUR SOLUTION
    # Note: this function assumpes that if the shape of x is four dimensional
    #       then we are trying to compute the fan size for a convolution layer's
    #       weights
    fan_in, fan_out = x.shape[-2:]
    if len(x.shape) == 4:
        # x is of shape (k, k, c_in, c_out)
        k2 = x.shape[0] * x.shape[1]
        fan_in *= float(k2)
        fan_out *= float(k2)
    return fan_in, fan_out
    ### END YOUR SOLUTION


def xavier_uniform(x, gain=1.0):
    ### BEGIN YOUR SOLUTION
    fan_in, fan_out = _calculate_fans(x)
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    uniform(x, low=-a, high=a)
    ### END YOUR SOLUTION


def xavier_normal(x, gain=1.0):
    ### BEGIN YOUR SOLUTION
    fan_in, fan_out = _calculate_fans(x)
    a = gain * math.sqrt(2 / (fan_in + fan_out))
    normal(x, std=a)
    ### END YOUR SOLUTION


def kaiming_uniform(x, mode="fan_in", nonlinearity="relu"):
    ### BEGIN YOUR SOLUTION
    fan_in, fan_out = _calculate_fans(x)
    fan_mode = fan_in if mode == "fan_in" else fan_out
    # only nonlinearity specified is ReLU
    # update this if we get other nonlinearities
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_mode)
    uniform(x, low=-bound, high=bound)
    ### END YOUR SOLUTION


def kaiming_normal(x, mode="fan_in", nonlinearity="relu"):
    ### BEGIN YOUR SOLUTION
    fan_in, fan_out = _calculate_fans(x)
    fan_mode = fan_in if mode == "fan_in" else fan_out
    # only nonlinearity specified is ReLU
    # update this if we get other nonlinearities
    gain = math.sqrt(2)
    bound = gain / math.sqrt(fan_mode)
    normal(x, std=bound)
    ### END YOUR SOLUTION
