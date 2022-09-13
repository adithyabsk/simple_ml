"""NDDArray backed computation backend.

This backend uses cuda backend_ndarray for cached data and computation.
"""
import numpy as np

import needle.device
from needle import backend_ndarray as nd
from needle.device import Device, DLDeviceType
from needle.ops import register_op_attr


class NDDevice(Device):
    def array(self, array, dtype):
        return nd.array(array, dtype=dtype, device=self.nd_device)

    def empty(self, shape, dtype):
        return nd.empty(shape, dtype=dtype, device=self.nd_device)

    def to_numpy(self, data):
        return data.numpy()

    def fill(self, array, fill_value):
        array.fill(fill_value)
        return array

    def randn(self, shape, dtype, mean=0.0, std=1.0):
        return nd.array(
            np.random.normal(loc=mean, scale=std, size=shape).astype(dtype),
            device=self.nd_device,
        )

    def randb(self, shape, dtype, ntrials=1, p=0.5):
        return nd.array(
            np.random.binomial(ntrials, p, size=shape).astype(dtype),
            device=self.nd_device,
        )

    def randu(self, shape, dtype, low=0, high=0):
        return nd.array(
            np.random.uniform(low=low, high=high, size=shape).astype(dtype),
            device=self.nd_device,
        )

    def one_hot(self, y, num_classes=10):
        # TODO fix this
        y_one_hot = []
        for i in range(y.shape[0]):
            y_one_hot.append(np.eye(num_classes)[int(y[i])])
        y_one_hot = np.array(y_one_hot)
        return nd.array(y_one_hot, device=self.nd_device)

    def enabled(self):
        return self.nd_device.enabled()

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.nd_compute(inputs, attrs)


class CUDADevice(NDDevice):
    def __init__(self, device_id: int = 0):
        assert device_id == 0
        self.nd_device = nd.cuda()
        self.device_id = device_id

    def __repr__(self):
        return "cuda(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CUDA, self.device_id)

    def __str__(self):
        return self.__repr__()


class CPUDevice(NDDevice):
    def __init__(self, device_id: int = 0):
        self.nd_device = nd.cpu()
        self.device_id = device_id

    def __repr__(self):
        return "cpu(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CPU, self.device_id)

    def __str__(self):
        return self.__repr__()


class OpenCLDevice(NDDevice):
    def __init__(self, device_id: int = 0):
        assert device_id == 0
        self.nd_device = nd.opencl()
        self.device_id = device_id

    def __repr__(self):
        return "opencl(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.OpenCL, self.device_id)

    def __str__(self):
        return self.__repr__()


def cuda(device_id: int = 0) -> CUDADevice:
    return CUDADevice(device_id)


def cpu() -> CPUDevice:
    return CPUDevice()


# set default device to be cpu device.
needle.device._DEFAULT_DEVICE = CPUDevice


def opencl(device_id: int = 0) -> CPUDevice:
    return OpenCLDevice(device_id)


def register_nd_compute(name, value=None):
    """Register the compute property based on backend_ndarray
    nd computation can be shared across multiple backends.
    """
    return register_op_attr(name, "nd_compute", value)


# device specific computations
@register_nd_compute("EWiseAdd")
def add(inputs, attrs):
    return inputs[0] + inputs[1]


@register_nd_compute("AddScalar")
def add_scalar(inputs, attrs):
    return inputs[0] + attrs["scalar"]


@register_nd_compute("EWiseMul")
def mul(inputs, attrs):
    return inputs[0] * inputs[1]


@register_nd_compute("MulScalar")
def mul_scalar(inputs, attrs):
    return inputs[0] * attrs["scalar"]


@register_nd_compute("EWiseDiv")
def divide(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] / inputs[1]
    ### END YOUR SOLUTION


@register_nd_compute("DivScalar")
def divide_scalar(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] / attrs["scalar"]
    ### END YOUR SOLUTION


@register_nd_compute("PowerScalar")
def power_scalar(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] ** attrs["scalar"]
    ### END YOUR SOLUTION


@register_nd_compute("MatMul")
def matmul(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] @ inputs[1]
    ### END YOUR SOLUTION


@register_nd_compute("Summation")
def summation(inputs, attrs):
    """
    Parameters:
    axes - int or tuple of ints or None

    If axes is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis.
    If axes is None, sum over all of the axes.

    Returns an array with the same shape, except with the specified axes removed.
    """
    ### BEGIN YOUR SOLUTION
    if attrs["axes"] is None:
        new_shape = tuple()
    elif isinstance(attrs["axes"], int):
        new_shape = list(inputs[0].shape)
        del new_shape[attrs["axes"]]
        new_shape = tuple(new_shape)
    else:  # axes is a tuple
        new_shape = tuple(
            s for i, s in enumerate(inputs[0].shape) if i not in attrs["axes"]
        )

    if attrs["axes"] is None or isinstance(attrs["axes"], int):
        return inputs[0].sum(axis=attrs["axes"]).reshape(new_shape)
    else:
        ret_arr = inputs[0].sum(axis=attrs["axes"][0])
        for a in attrs["axes"][1:]:
            ret_arr = ret_arr.sum(a)
        return ret_arr.reshape(new_shape)
    ### END YOUR SOLUTION


@register_nd_compute("BroadcastTo")
def broadcast_to(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].broadcast_to(new_shape=attrs["shape"])
    ### END YOUR SOLUTION


@register_nd_compute("Reshape")
def reshape(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].reshape(new_shape=attrs["shape"])
    ### END YOUR SOLUTION


@register_nd_compute("Permute")
def permute(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].permute(new_axes=attrs["new_axes"])
    ### END YOUR SOLUTION


@register_nd_compute("Negate")
def negate(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] * -1
    ### END YOUR SOLUTION


@register_nd_compute("Transpose")
def transpose(inputs, attrs):
    """
    Parameters:
    axes - tuple of ints or None

    If axes is a tuple of ints, permute those two axes.
    If axes is None, permutes the last two axes.
    """
    ### BEGIN YOUR SOLUTION
    shape_size = len(inputs[0].shape)
    axes = attrs.get("axes", None)
    if axes is None:
        axes = (shape_size - 2, shape_size - 1)
    # generate the permutation template
    new_axes = list(range(shape_size))
    # swap axes
    new_axes[axes[1]], new_axes[axes[0]] = new_axes[axes[0]], new_axes[axes[1]]
    new_axes = tuple(new_axes)
    return inputs[0].permute(new_axes=new_axes)
    ### END YOUR SOLUTION


@register_nd_compute("Log")
def log(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].log()
    ### END YOUR SOLUTION


@register_nd_compute("Exp")
def exp(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].exp()
    ### END YOUR SOLUTION


@register_nd_compute("ReLU")
def relu(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].maximum(0)
    ### END YOUR SOLUTION


@register_nd_compute("LogSoftmax")
def logsoftmax(inputs, attrs):
    """
    Computes log softmax along the last dimension of the array.
    """
    ### BEGIN YOUR SOLUTION
    shape_size = len(inputs[0].shape)
    last_axis = shape_size - 1
    # in the next two lines we need to reshape followed by a broadcast to get
    # the dimensions to match the input dimensions. The proper solution here is
    # to implement a keep_dims argument but I'm leaving that as a todo for
    # future work
    x = inputs[0] - inputs[0].max(axis=last_axis).reshape(
        (inputs[0].shape[:-1]) + (1,)
    ).broadcast_to(new_shape=inputs[0].shape)
    return (
        x
        - x.exp()
        .sum(axis=last_axis)
        .reshape((inputs[0].shape[:-1]) + (1,))
        .broadcast_to(new_shape=inputs[0].shape)
        .log()
    )
    ### END YOUR SOLUTION


@register_nd_compute("Tanh")
def tanh(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].tanh()
    ### END YOUR SOLUTION


@register_nd_compute("GetItem")
def get_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Returns array indexed by idxs i.e. if array A has shape (5, 3, 2),
    then the shape of the A[0, :, :] would be (3, 2).
    """
    ### BEGIN YOUR SOLUTION
    return inputs[0][attrs["idxs"]]
    ### END YOUR SOLUTION


@register_nd_compute("SetItem")
def set_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Sets array A at idxs with array B and returns the array.
    """
    ### BEGIN YOUR SOLUTION
    inputs[0][attrs["idxs"]] = inputs[1]
    return inputs[0]
    ### END YOUR SOLUTION


@register_nd_compute("Stack")
def stack(As, attrs):
    """
    Concatenates a sequence of arrays along a new dimension.

    Parameters:
    axis - dimension to concatenate along

    All arrays need to be of the same size.
    """
    ### BEGIN YOUR SOLUTION
    axis = attrs["axis"]
    orig_shape = As[0].shape
    init_shape = orig_shape[:axis] + (1,) + orig_shape[axis:]
    As = [arr.reshape(init_shape) for arr in As]
    out_shape = list(init_shape)
    out_shape[axis] = sum([arr.shape[axis] for arr in As])
    out_arr = nd.empty(tuple(out_shape), device=As[0].device)
    # create a list of empty slices
    slices = [slice(None) for i in range(len(init_shape))]
    start = 0
    for i, arr in enumerate(As):
        if len(arr.shape) != len(init_shape):
            raise ValueError(
                f"Input array {i} does not have the same shape as the first" "array"
            )
        if not all(
            s == is_
            for i, (s, is_) in enumerate(zip(arr.shape, init_shape))
            if i != axis
        ):
            raise ValueError(
                f"Input array {i} shape dimensions do not match the first"
                f"array's shape dimensions (except for axis {axis})"
            )
        slices[axis] = slice(start, start + 1)
        out_arr[tuple(slices)] = arr
        start += 1

    return out_arr
    ### END YOUR SOLUTION


@register_nd_compute("Flip")
def flip(inputs, attrs):
    """
    Flips the input along specified axes.

    Parameters:
    axes - Axes to flip.
    """
    ### BEGIN YOUR SOLUTION
    axes = attrs["axes"]
    return inputs[0].flip(axes)
    ### END YOUR SOLUTION


@register_nd_compute("Pad")
def pad(inputs, attrs):
    """
    Pad the input along specified axes.

    Parameters:
    axes - Axes to flip.
    """
    ### BEGIN YOUR SOLUTION
    axes = attrs["axes"]
    return inputs[0].pad(axes)
    ### END YOUR SOLUTION


@register_nd_compute("Dilate")
def dilate(inputs, attrs):
    """
    Dilates the input by a dilation factor on specified axes.
    (i.e., inserts 0s between elements)

    Parameters:
    dilation - Dilation amount (number of 0s to insert)
    axes - Axes to dilate by this amount
    """
    ### BEGIN YOUR SOLUTION
    dilation = attrs["dilation"]
    axes = attrs["axes"]
    a = inputs[0]
    if axes is None:
        axes = range(len(a.shape))
    out_shape = [s * (dilation + 1) if i in axes else s for i, s in enumerate(a.shape)]
    out = nd.empty(out_shape, device=a.device)
    out.fill(0.0)
    slices = [
        slice(None, None, dilation + 1) if i in axes else slice(None)
        for i, s in enumerate(a.shape)
    ]
    out[tuple(slices)] = a
    return out
    ### END YOUR SOLUTION


@register_nd_compute("Conv")
def conv(inputs, attrs):
    """
    Multi-channel 2D convolution of two inputs (called input and weight, respectively).
    inputs[0]: "input", NHWC
    inputs[1]: "weight", (kernel_size, kernel_size, c_in, c_out)

    Parameters:
    padding - (int) Pad the HW axes of the input by this amount
    stride - (int) Stride of the convolution
    """
    ### BEGIN YOUR SOLUTION
    padding = attrs["padding"]
    stride = attrs["stride"]
    tensor = inputs[0]
    weight = inputs[1]
    pad_axes = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    Z = tensor.pad(axes=pad_axes)
    N, H, W, C_in = Z.shape
    K, K_, C_in_, C_out = weight.shape
    if K != K_:
        raise ValueError(f"weight kernel sizes must be equal ({K} != {K_})")
    if C_in_ != C_in:
        raise ValueError(f"channel in must match ({C_in} != {C_in_})")

    Ns, Hs, Ws, Cs = Z.strides
    inner_dim = K * K * C_in
    # import pdb; pdb.set_trace()
    A = nd.NDArray.make(
        (N, (H - K) // stride + 1, (W - K) // stride + 1, K, K, C_in),
        strides=(Ns, Hs * stride, Ws * stride, Hs, Ws, Cs),
        device=Z.device,
        handle=Z._handle,
    ).reshape((N * ((H - K) // stride + 1) * ((W - K) // stride + 1), inner_dim))
    out = A @ weight.reshape((K * K * C_in, C_out))
    return out.reshape((N, (H - K) // stride + 1, (W - K) // stride + 1, C_out))
    ### END YOUR SOLUTION


@register_nd_compute("Conv4")
def conv4(inputs, attrs):
    padding = attrs["padding"]
    stride = attrs["stride"]
    if stride is None:
        stride = 1
    tensor = inputs[0]
    weight = inputs[1]
    pad_axes = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    Z = tensor.pad(axes=pad_axes)
    N, H, W, C_in = Z.shape
    K, K_, C_in_, C_out = weight.shape
    if K != K_:
        raise ValueError(f"weight kernel sizes must be equal ({K} != {K_})")
    if C_in_ != C_in:
        raise ValueError(f"channel in must match ({C_in} != {C_in_})")

    Ns, Hs, Ws, Cs = Z.strides

    if not hasattr(tensor, "conv4"):
        raise ValueError("tensor does not have c++ conv implementation")
    return Z.conv4(weight, stride)
    ### END YOUR SOLUTION
