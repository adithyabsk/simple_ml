"""The module."""
from __future__ import annotations

import math
import operator
from functools import reduce
from typing import List, Tuple

import needle.init as init
import numpy as np
from needle import ops
from needle.autograd import Tensor


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for v in value.values():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    ### BEGIN YOUR SOLUTION
    if isinstance(value, Module):
        return [value] + _child_modules(value._children())
    elif isinstance(value, dict):
        params = []
        for v in value.values():
            params += _child_modules(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _child_modules(v)
        return params
    else:
        return []
    ### END YOUR SOLUTION


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        ### BEGIN YOUR SOLUTION
        root_k = math.sqrt(1 / in_features)
        self.weight = Parameter(
            ops.zeros(shape=(in_features, out_features), dtype=dtype, device=device)
        )
        init.uniform(self.weight, low=-root_k, high=root_k)
        if bias:
            self.bias = (
                None
                if not bias
                else Parameter(
                    ops.zeros(shape=(out_features,), dtype=dtype, device=device)
                )
            )
            init.uniform(self.bias, low=-root_k, high=root_k)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x @ self.weight
        if self.bias is not None:
            bias_shape = [1] * len(y.shape)
            bias_shape[-1] = self.bias.shape[0]
            bias_shape = tuple(bias_shape)
            bias = self.bias.reshape(bias_shape).broadcast_to(y.shape)
            y = y + bias
        return y
        ### END YOUR SOLUTION


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (ops.exp(-x) + 1.0) ** (-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return reduce(
            lambda layer_input, layer: layer.forward(layer_input), [x, *self.modules]
        )
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n_classes = x.shape[-1]
        sm = -(
            ops.one_hot(y, num_classes=n_classes, device=x.device, dtype=x.dtype)
            * ops.logsoftmax(x)
        )
        last_dim = len(sm.shape) - 1
        sm = sm.sum(axes=last_dim)
        numer = sm.sum(axes=0)

        return numer / sm.size
        ### END YOUR SOLUTION


class BatchNorm(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(ops.ones(shape=(dim,), dtype=dtype, device=device))
        self.bias = Parameter(ops.zeros(shape=(dim,), dtype=dtype, device=device))
        self.running_mean = ops.zeros(shape=(dim,), dtype=dtype, device=device)
        self.running_var = ops.ones(shape=(dim,), dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def get_dims_and_sum_dims(self, x: Tensor) -> Tuple:
        dims = list(x.shape)
        sum_dims = list(range(len(dims)))
        del dims[1]
        del sum_dims[1]  # drop the first dim
        return tuple(dims), tuple(sum_dims)

    def expectation(self, x: Tensor) -> Tensor:
        dims, sum_dims = self.get_dims_and_sum_dims(x)
        div_size = reduce(operator.mul, dims)

        ret_shape = [1] * len(x.shape)
        ret_shape[1] = x.shape[1]
        ret_shape = tuple(ret_shape)

        return (x.sum(axes=sum_dims) / div_size).reshape(ret_shape)

    def variance(self, exp_val, x: Tensor) -> Tensor:
        dims, sum_dims = self.get_dims_and_sum_dims(x)
        div_size = reduce(operator.mul, dims)

        ret_shape = [1] * len(x.shape)
        ret_shape[1] = x.shape[1]
        ret_shape = tuple(ret_shape)

        return (
            ((x - self.match_output_shape(exp_val, x.shape)) ** 2).sum(axes=sum_dims)
            / (div_size - 1)
        ).reshape(ret_shape)

    def match_output_shape(self, x: Tensor, shape: Tuple) -> Tensor:
        new_shape = (1,) + (self.dim,) + (len(shape) - 2) * (1,)
        return ops.broadcast_to(ops.reshape(x, shape=new_shape), shape=shape)

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            exp_val = self.expectation(x)
            # E((x-E(x))^2)
            var_val_unbiased = self.expectation(
                (x - ops.broadcast_to(self.expectation(x), shape=x.shape)) ** 2
            )
            var_val_biased = self.variance(exp_val, x)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * exp_val.reshape(
                self.running_mean.shape
            )
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var_val_biased.reshape(
                self.running_var.shape
            )

            numer = x - self.match_output_shape(exp_val, shape=x.shape)
            numer = self.match_output_shape(self.weight, shape=numer.shape) * numer
            denom = (self.eps + var_val_unbiased) ** (1 / 2)
            y = numer / self.match_output_shape(denom, shape=numer.shape)
        else:
            numer = x - self.match_output_shape(self.running_mean, shape=x.shape)
            numer = self.match_output_shape(self.weight, shape=numer.shape) * numer
            denom = (self.eps + self.running_var) ** (1 / 2)
            y = numer / self.match_output_shape(denom, shape=numer.shape)

        # we multiply the weight above because of a numerical stability quirk
        # in the grading
        # https://forum.dlsyscourse.org/t/tiny-numerical-error-in-batchnorm/417/2?u=adithya
        y = y + self.match_output_shape(self.bias, shape=y.shape)
        return y
        ### END YOUR SOLUTION


class LayerNorm(Module):
    def __init__(self, dims, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(ops.ones(shape=self.dims, dtype=dtype, device=device))
        self.bias = Parameter(ops.zeros(shape=self.dims, dtype=dtype, device=device))
        ### END YOUR SOLUTION

    def expectation(self, x: Tensor) -> Tensor:
        dim_len = len(self.dims)
        sum_dims = tuple(range(len(x.shape)))[-dim_len:]
        new_shape = x.shape[:-dim_len] + (1,) * dim_len
        div_size = reduce(operator.mul, self.dims)
        # the reshape and broadcast gets x back to its original shape
        return (
            (x.sum(axes=sum_dims) / div_size).reshape(new_shape).broadcast_to(x.shape)
        )

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        exp_val = self.expectation(x)
        # E((x-E(x))^2)
        var_val = self.expectation((x - self.expectation(x)) ** 2)
        numer = x - exp_val
        denom = (self.eps + var_val) ** (1 / 2)
        y = numer / denom

        new_shape = (1,) * (len(y.shape) - len(self.dims)) + self.weight.shape
        y = y * self.weight.reshape(new_shape).broadcast_to(y.shape)
        y = y + self.bias.reshape(new_shape).broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, drop_prob, device=None, dtype="float32"):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_mat = ops.randb(x.shape, p=(1 - self.p), device=x.device)
            ret_arr = x * drop_mat
            ret_arr /= 1 - self.p
            return ret_arr
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module, device=None, dtype="float32"):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Flatten(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format

    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    conv_op = ops.conv

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            ops.zeros(
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                dtype=dtype,
                device=device,
            )
        )
        init.kaiming_uniform(self.weight)
        if bias:
            bias_interval = 1.0 / ((in_channels * kernel_size ** 2) ** 0.5)
            self.bias = Parameter(
                ops.zeros(shape=(out_channels,), dtype=dtype, device=device)
            )
            init.uniform(self.bias, low=-bias_interval, high=bias_interval)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # NCHW --> NHWC
        x = x.permute((0, 2, 3, 1))
        # then we pad to make sure that input == output
        # conv output shape is (n, (h-k)//stride + 1, (w-k)//stride + 1, c_out)
        _, H, W, _ = x.shape
        if H != W:
            raise ValueError("Cannot handle non-square inputs")
        padding = (self.kernel_size - 1) // 2
        y = self.conv_op(x, self.weight, padding=padding, stride=self.stride)
        if self.bias is not None:
            bias_shape = [1] * len(y.shape)
            bias_shape[-1] = self.bias.shape[0]
            y = y + self.bias.reshape(tuple(bias_shape)).broadcast_to(shape=y.shape)
        # NHWC --> NCHW
        y = y.permute((0, 3, 1, 2))
        return y
        ### END YOUR SOLUTION


class Conv4(Conv):
    conv_op = ops.conv4


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.W_ih = Parameter(
            ops.zeros(shape=(input_size, hidden_size), dtype=dtype, device=device)
        )
        self.W_hh = Parameter(
            ops.zeros(shape=(hidden_size, hidden_size), dtype=dtype, device=device)
        )
        sqrt_k = (1.0 / hidden_size) ** (1 / 2)
        init.uniform(self.W_ih, low=-sqrt_k, high=sqrt_k)
        init.uniform(self.W_hh, low=-sqrt_k, high=sqrt_k)
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(
                ops.zeros(shape=(hidden_size,), dtype=dtype, device=device)
            )
            self.bias_hh = Parameter(
                ops.zeros(shape=(hidden_size,), dtype=dtype, device=device)
            )
            init.uniform(self.bias_ih, low=-sqrt_k, high=sqrt_k)
            init.uniform(self.bias_hh, low=-sqrt_k, high=sqrt_k)
        self.act = ReLU() if nonlinearity == "relu" else Tanh()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = ops.zeros(
                shape=(X.shape[0], self.hidden_size),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
        h_prime = X @ self.W_ih

        if self.bias_ih is not None:
            bias_shape = [1] * len(h_prime.shape)
            bias_shape[-1] = self.bias_ih.shape[0]
            bias_shape = tuple(bias_shape)
            bias_ih = self.bias_ih.reshape(bias_shape).broadcast_to(h_prime.shape)
            h_prime = h_prime + bias_ih

        h_prime = h_prime + h @ self.W_hh

        if self.bias_hh is not None:
            bias_shape = [1] * len(h_prime.shape)
            bias_shape[-1] = self.bias_hh.shape[0]
            bias_shape = tuple(bias_shape)
            bias_hh = self.bias_hh.reshape(bias_shape).broadcast_to(h_prime.shape)
            h_prime = h_prime + bias_hh

        return self.act(h_prime)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            if i == 0
            else RNNCell(
                hidden_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        outputs = []
        h_n = []
        seq_len = X.shape[0]
        prev_h_list = [None] * self.num_layers
        X0_slice_shape = tuple(X.shape[1:])
        if h0 is not None:
            other_dims = tuple([slice(None) for _ in range(len(h0.shape) - 1)])
            h0_slice_shape = h0.shape[1:]
            prev_h_list = [
                h0[(i,) + other_dims].reshape(h0_slice_shape)
                for i in range(h0.shape[0])
            ]
        for i in range(seq_len):
            slices = tuple([i] + [slice(None) for _ in range(len(X.shape) - 1)])
            # the reshape is to handle the degenerate case where we end up with
            # a scalar and need to get back to multiple dimensions
            X_i = X[slices].reshape(X0_slice_shape)
            curr_hidden = []
            for prev_h, cell in zip(prev_h_list, self.rnn_cells):
                h_t = cell(X_i, prev_h)
                curr_hidden.append(h_t)
                if i == (seq_len - 1):
                    h_n.append(h_t)
                X_i = h_t
            prev_h_list = curr_hidden
            outputs.append(h_t)
        stacked_output = ops.stack(outputs, axis=0)
        stacked_h_n = ops.stack(h_n, axis=0)

        return stacked_output, stacked_h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bias = bias
        self.dtype = dtype
        self.device = device
        self.W_ih = Parameter(
            ops.zeros(shape=(input_size, 4 * hidden_size), dtype=dtype, device=device)
        )
        self.W_hh = Parameter(
            ops.zeros(shape=(hidden_size, 4 * hidden_size), dtype=dtype, device=device)
        )
        sqrt_k = (1.0 / hidden_size) ** (1 / 2)
        init.uniform(self.W_ih, low=-sqrt_k, high=sqrt_k)
        init.uniform(self.W_hh, low=-sqrt_k, high=sqrt_k)
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(
                ops.zeros(shape=(4 * hidden_size,), dtype=dtype, device=device)
            )
            self.bias_hh = Parameter(
                ops.zeros(shape=(4 * hidden_size,), dtype=dtype, device=device)
            )
            init.uniform(self.bias_ih, low=-sqrt_k, high=sqrt_k)
            init.uniform(self.bias_hh, low=-sqrt_k, high=sqrt_k)
        ### END YOUR SOLUTION

    def _inner_expression(self, X, h, W_ix, W_hx, b_ix, b_hx):
        inner_exp = X @ W_ix

        if self.bias:
            bias_shape = [1] * len(inner_exp.shape)
            bias_shape[-1] = b_ix.shape[0]
            bias_shape = tuple(bias_shape)
            bias_ix = b_ix.reshape(bias_shape).broadcast_to(inner_exp.shape)
            inner_exp = inner_exp + bias_ix

        inner_exp = inner_exp + h @ W_hx

        if self.bias:
            bias_shape = [1] * len(inner_exp.shape)
            bias_shape[-1] = b_hx.shape[0]
            bias_shape = tuple(bias_shape)
            bias_hx = b_hx.reshape(bias_shape).broadcast_to(inner_exp.shape)
            inner_exp = inner_exp + bias_hx

        return inner_exp

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h0 = ops.zeros(
                shape=(X.shape[0], self.hidden_size),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            c0 = ops.zeros(
                shape=(X.shape[0], self.hidden_size),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
        else:
            h0, c0 = h

        hs = self.hidden_size
        W_ii, W_if, W_ig, W_io = [
            self.W_ih[(slice(None), slice(i * hs, i * hs + hs))].reshape(
                (self.input_size, hs)
            )
            for i in range(4)
        ]
        W_hi, W_hf, W_hg, W_ho = [
            self.W_hh[(slice(None), slice(i * hs, i * hs + hs))].reshape(
                (self.hidden_size, hs)
            )
            for i in range(4)
        ]
        b_ii, b_if, b_ig, b_io = [None] * 4
        b_hi, b_hf, b_hg, b_ho = [None] * 4
        if self.bias:
            b_ii, b_if, b_ig, b_io = [
                self.bias_ih[slice(i * hs, i * hs + hs)].reshape((hs,))
                for i in range(4)
            ]
            b_hi, b_hf, b_hg, b_ho = [
                self.bias_hh[slice(i * hs, i * hs + hs)].reshape((hs,))
                for i in range(4)
            ]

        i = Sigmoid()(self._inner_expression(X, h0, W_ii, W_hi, b_ii, b_hi))
        f = Sigmoid()(self._inner_expression(X, h0, W_if, W_hf, b_if, b_hf))
        g = Tanh()(self._inner_expression(X, h0, W_ig, W_hg, b_ig, b_hg))
        o = Sigmoid()(self._inner_expression(X, h0, W_io, W_ho, b_io, b_ho))
        c_prime = f * c0 + i * g
        h_prime = o * Tanh()(c_prime)

        return (h_prime, c_prime)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cells = [
            LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)
            if i == 0
            else LSTMCell(
                hidden_size, hidden_size, bias=bias, device=device, dtype=dtype
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        outputs = []
        h_n = []
        c_n = []
        seq_len = X.shape[0]
        prev_h_list = [None] * self.num_layers
        X0_slice_shape = tuple(X.shape[1:])
        if h is not None:
            h0, c0 = h
            # note: h0.shape == c0.shape
            other_dims = tuple([slice(None) for _ in range(len(h0.shape) - 1)])
            h0_slice_shape = h0.shape[1:]
            prev_h_list = [
                (
                    h0[(i,) + other_dims].reshape(h0_slice_shape),
                    c0[(i,) + other_dims].reshape(h0_slice_shape),
                )
                for i in range(h0.shape[0])
            ]
        for i in range(seq_len):
            slices = tuple([i] + [slice(None) for _ in range(len(X.shape) - 1)])
            # the reshape is to handle the degenerate case where we end up with
            # a scalar and need to get back to multiple dimensions
            X_i = X[slices].reshape(X0_slice_shape)
            curr_hidden = []
            for prev_h, lstm_cell in zip(prev_h_list, self.lstm_cells):
                # note h_t is a tuple (h_prime, c_prime)
                h_t = lstm_cell(X_i, prev_h)
                curr_hidden.append(h_t)
                if i == (seq_len - 1):
                    h_n.append(h_t[0])
                    c_n.append(h_t[1])
                X_i = h_t[0]
            prev_h_list = curr_hidden
            outputs.append(h_t[0])
        stacked_output = ops.stack(outputs, axis=0)
        stacked_h_n = ops.stack(h_n, axis=0)
        stacked_c_n = ops.stack(c_n, axis=0)

        return stacked_output, (stacked_h_n, stacked_c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.device = device
        self.weight = Parameter(
            ops.zeros(shape=(num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        init.normal(self.weight)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        hot_stack = []
        for s in range(x.shape[0]):
            hot_stack.append(
                ops.one_hot(
                    x[s, :].reshape((x.shape[1],)),
                    num_classes=self.num_embeddings,
                    dtype=self.dtype,
                    device=self.device,
                )
                @ self.weight
            )
        x_one_hot = ops.stack(hot_stack, axis=0).reshape(
            (x.shape[0], x.shape[1], self.embedding_dim)
        )

        return x_one_hot
        ### END YOUR SOLUTION
