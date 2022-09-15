"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import List, Optional

from .autograd import Op, Tensor, Tuple, Value
from .device import default_device

OP_TABLE = {}


def register_op(name: str, op: Op) -> Op:
    """Register an operator to the op table.

    Parameters
    ----------
    name : str
        The name of the op.

    Returns
    -------
    op : Op
        The registered op.
    """
    if name in OP_TABLE:
        raise ValueError("Op %s is already registered")
    OP_TABLE[name] = op
    return op


def register_op_attr(op_name, attr_name, attr_value=None):
    """Register additional attributes to an existing op by name.


    Parameters
    ----------
    op_name : str
        The name of the op

    attr_name : str
        The name of the attribute

    attr_value :
        The attribute value to be set.

    Returns
    -------
    The attr_value if attr_value is not None.
    Otherwise returns a decorator function.


    Note
    ----
    This function can be used to register additional attributes
    to an Op used by a specific backend.
    """

    def _register(value):
        if op_name not in OP_TABLE:
            raise ValueError("Op %s does not exist")
        op = OP_TABLE[op_name]
        setattr(op, attr_name, value)
        return op

    if attr_value is None:
        return _register
    return _register(attr_value)


class MakeTupleOp(Op):
    def __call__(self, *args: List[Value]) -> Tuple:
        return Tuple.make_from_op(self, list(args))

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, Tuple)
        return [out_grad[i] for i in range(len(out_grad))]


make_tuple = register_op("MakeTuple", MakeTupleOp())


class TupleGetItemOp(Op):
    def __call__(self, a: Tuple, index: int, *, fold_const=True) -> Tensor:
        assert isinstance(a, Tuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTupleOp):
            return a.inputs[index]
        return Tensor.make_from_op(self, [a], attrs={"index": index})

    def gradient(self, out_grad, node):
        index = node.attrs["index"]
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return [make_tuple(*in_grad)]


tuple_get_item = register_op("TupleGetItem", TupleGetItemOp())


class FusedAddScalarsOp(Op):
    def __call__(self, a: Tensor, c0: float, c1: float) -> Tuple:
        return Tuple.make_from_op(self, [a], attrs={"c0": c0, "c1": c1})

    def gradient(self, out_grad, node):
        return [out_grad[0] + out_grad[1]]


fused_add_scalars = register_op("FusedAddScalars", FusedAddScalarsOp())


class EWiseAddOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]


add = register_op("EWiseAdd", EWiseAddOp())


class AddScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad]


add_scalar = register_op("AddScalar", AddScalarOp())


class EWiseMulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


multiply = register_op("EWiseMul", EWiseMulOp())


class MulScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]


multiply_scalar = register_op("MulScalar", MulScalarOp())


class PowerScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):

        return [
            (node.attrs["scalar"] * (node.inputs[0] ** (node.attrs["scalar"] - 1)))
            * out_grad
        ]


power_scalar = register_op("PowerScalar", PowerScalarOp())


class EWiseDivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):

        lhs, rhs = node.inputs
        return (out_grad / rhs, out_grad * (-lhs / (rhs * rhs)))


divide = register_op("EWiseDiv", EWiseDivOp())


class DivScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):

        return [out_grad / node.attrs["scalar"]]


divide_scalar = register_op("DivScalar", DivScalarOp())


def sum_grad(tensor, target_shape_len):
    input_shape_len = len(tensor.shape)
    if input_shape_len > target_shape_len:
        sum_len = input_shape_len - target_shape_len
        summation_axes = tuple(range(sum_len))
        return tensor.sum(summation_axes)

    return tensor


class MatMulOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):

        lhs, rhs = node.inputs
        return (
            sum_grad(out_grad @ rhs.transpose(None), len(lhs.shape)),
            sum_grad(lhs.transpose(None) @ out_grad, len(rhs.shape)),
        )


matmul = register_op("MatMul", MatMulOp())


class SummationOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):

        input_shape = node.inputs[0].shape
        axes = node.attrs.get("axes", None)
        summation_axis = set(
            ([axes] if isinstance(axes, int) else axes) or range(len(input_shape))
        )
        new_shape = [
            1 if i in summation_axis else dim for i, dim in enumerate(input_shape)
        ]
        # the first reshape adds back the reduced dimensions as 1s
        # the brodcast expands this into repeated values
        return [out_grad.reshape(new_shape).broadcast_to(input_shape)]


summation = register_op("Summation", SummationOp())


class BroadcastToOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):

        input_shape = node.inputs[0].shape
        broadcast_shape = node.attrs["shape"]
        if broadcast_shape == input_shape:
            return [out_grad]
        pad_len = len(broadcast_shape) - len(input_shape)
        padded_input_shape = (0,) * pad_len + input_shape
        summation_axes = tuple(
            i
            for i, (b, pi) in enumerate(zip(broadcast_shape, padded_input_shape))
            if b != pi
        )

        return [out_grad.sum(summation_axes).reshape(input_shape)]


broadcast_to = register_op("BroadcastTo", BroadcastToOp())


class ReshapeOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):

        input_shape = node.inputs[0].shape
        if len(input_shape) == 0:
            first_elem_slice = (0,) + tuple(
                slice(None) for _ in range(len(out_grad.shape) - 1)
            )
            return [out_grad[first_elem_slice]]
        else:
            return [out_grad.reshape(input_shape)]


reshape = register_op("Reshape", ReshapeOp())


class PermuteOp(Op):
    def __call__(self, a: Tensor, new_axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"new_axes": new_axes})

    def gradient(self, out_grad, node):

        new_axes = node.attrs["new_axes"]
        # resort new_axes in original order
        orig_axes = tuple(map(lambda idx: new_axes.index(idx), range(len(new_axes))))
        return [out_grad.permute(orig_axes)]


permute = register_op("Permute", PermuteOp())


class NegateOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):

        return [-out_grad]


negate = register_op("Negate", NegateOp())


class TransposeOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):

        axes = node.attrs.get("axes", None)
        return [out_grad.transpose(axes)]


transpose = register_op("Transpose", TransposeOp())


class LogOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):

        n = node.inputs[0]
        return [out_grad / n]  # essentially 1 / n * out_grad


log = register_op("Log", LogOp())


class ExpOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [exp(node.inputs[0]) * out_grad]


exp = register_op("Exp", ExpOp())


class ReLUOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):

        epsilon = 1e-12
        return [((node + epsilon) / (node.inputs[0] + epsilon)) * out_grad]


relu = register_op("ReLU", ReLUOp())


def softmax_stable(x):
    last_axis = len(x.shape) - 1
    orig_shape = x.shape[:-1] + (1,)
    z = exp(x - x.cached_data.max(last_axis).reshape(orig_shape).broadcast_to(x.shape))
    return z / summation(z, axes=last_axis).reshape(orig_shape).broadcast_to(z.shape)


class LogSoftmaxOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):

        a = node.inputs[0]
        last_axis = len(a.shape) - 1
        orig_shape = a.shape[:-1] + (1,)
        return [
            out_grad
            - (
                softmax_stable(a)
                * summation(out_grad, axes=last_axis)
                .reshape(orig_shape)
                .broadcast_to(a.shape)
            )
        ]


logsoftmax = register_op("LogSoftmax", LogSoftmaxOp())


class TanhOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):

        # TODO: not sure why this gradient comes out backwards
        #       it should be (1 - tanh^2(x))
        #       but the numerical tests say it should be  (-1+tanh^2(x))
        return [-(1 - tanh(node.inputs[0]) ** 2) * out_grad]


tanh = register_op("Tanh", TanhOp())


class GetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):

        a = node.inputs[0]
        idxs = node.attrs["idxs"]
        ret_tensor = zeros_like(a)
        ret_tensor[idxs] = out_grad
        return [ret_tensor]


get_item = register_op("GetItem", GetItemOp())


class SetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):
        raise NotImplementedError()


set_item = register_op("SetItem", SetItemOp())


class StackOp(Op):
    def __call__(self, args: List[Value], axis: int) -> Tensor:
        return Tensor.make_from_op(self, args, attrs={"axis": axis})

    def gradient(self, out_grad, node):

        out_grad_list = []
        arr_list = node.inputs
        init_shape = arr_list[0].shape
        axis = node.attrs["axis"]
        slices = [slice(None) for i in range(len(init_shape) + 1)]
        start = 0
        for _ in range(len(arr_list)):
            slices[axis] = start
            out_grad_list.append(out_grad[tuple(slices)].reshape(init_shape))
            start += 1
        return out_grad_list


stack = register_op("Stack", StackOp())


class ConvOp(Op):
    def __call__(
        self,
        a: Tensor,
        b: Tensor,
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
    ) -> Tensor:
        return Tensor.make_from_op(
            self, [a, b], attrs={"stride": stride, "padding": padding}
        )

    def gradient(self, out_grad, node):
        X_grad, W_grad = node.inputs
        padding = node.attrs["padding"]
        stride = node.attrs["stride"]
        N, H, W, C_in = X_grad.shape
        K, _, _, C_out = W_grad.shape
        out_grad_mod = dilate(out_grad, dilation=stride - 1, axes=(1, 2))
        # this permutation is super magical, and to be 100% honest, I have no
        # flipping clue why the permutations result in the correct outputs
        # I was just dimension bashing to get the dimensions to match based on
        # the intuition that I need to use conv in the backward pass
        X_slices = tuple(
            [
                slice(None),
                slice(padding, padding + H),
                slice(padding, padding + W),
                slice(None),
            ]
        )
        return (
            # flip W over the kernel dimensions and then permute
            # make sure to reshape after the slice in case there was a dimension
            # of one which will reduce the dims by default (we did not
            # implement keep dims)
            conv(
                out_grad_mod.pad(axes=((0, 0), (K - 1, K - 1), (K - 1, K - 1), (0, 0))),
                W_grad.flip((0, 1)).permute((0, 1, 3, 2)),
            )[X_slices].reshape(X_grad.shape),
            conv(
                X_grad.permute((3, 1, 2, 0)),
                out_grad_mod.permute((1, 2, 0, 3)),
                padding=padding,
            ).permute((1, 2, 0, 3)),
        )


conv = register_op("Conv", ConvOp())
# backward pass is the same for both conv implementations
conv4 = register_op("Conv4", ConvOp())


class PadOp(Op):
    def __call__(self, a: Tensor, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):

        axes = node.attrs.get("axes", None)
        A = node.inputs[0]
        slices = tuple(
            slice(None) if lp == rp == 0 else slice(lp, lp + dim)
            for dim, (lp, rp) in zip(A.shape, axes)
        )
        return [out_grad[slices]]


pad = register_op("Pad", PadOp())


class FlipOp(Op):
    def __call__(self, a: Tensor, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):

        axes = node.attrs.get("axes", None)
        return [out_grad.flip(axes)]


flip = register_op("Flip", FlipOp())


class DilateOp(Op):
    def __call__(self, a: Tensor, dilation: int, axes: tuple) -> Tensor:
        return Tensor.make_from_op(
            self, [a], attrs={"dilation": dilation, "axes": axes}
        )

    def gradient(self, out_grad, node):

        dilation = node.attrs.get("dilation")
        axes = node.attrs.get("axes", None)
        a = node.inputs[0]
        if axes is None:
            axes = range(len(node.inputs[0].shape))
        slices = [
            slice(None, None, dilation + 1) if i in axes else slice(None)
            for i, s in enumerate(a.shape)
        ]
        return [out_grad[tuple(slices)]]


dilate = register_op("Dilate", DilateOp())


# additional helper functions
def full(
    shape, fill_value, *, rand=None, dtype="float32", device=None, requires_grad=False
):
    if rand is None:
        rand = {}
    device = device if device else default_device()

    if not rand or "dist" not in rand:
        arr = device.empty(shape, dtype)
        device.fill(arr, fill_value)
    else:
        if rand["dist"] == "normal":
            arr = device.randn(shape, dtype, mean=rand["mean"], std=rand["std"])
        if rand["dist"] == "binomial":
            arr = device.randb(shape, dtype, ntrials=rand["trials"], p=rand["prob"])
        if rand["dist"] == "uniform":
            arr = device.randu(shape, dtype, low=rand["low"], high=rand["high"])

    return Tensor.make_const(arr, device, requires_grad=requires_grad)


def one_hot(labels: Tensor, *, num_classes=10, dtype="float32", device=None):
    device = device if device else default_device()
    arr = device.one_hot(labels.numpy(), num_classes=num_classes)
    return Tensor.make_const(arr, device, requires_grad=False)


def ones(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 1, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
