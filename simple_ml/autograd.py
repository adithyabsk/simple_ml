"""Core data structures."""
from enum import Enum
from typing import Dict, List, Optional

import simple_ml

from .device import CachedData, Device, default_device

LAZY_MODE = False
TENSOR_COUNTER = 0


class Op:
    """Operator definition."""

    def gradient(self, out_grad: "Value", node: "Value") -> List["Value"]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: List[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    attrs: object
    # The following fields are cached fields for
    # dynamic computation
    cached_data: CachedData
    cached_device: Device
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.cached_device.compute(
            self.op, [x.realize_cached_data() for x in self.inputs], self.attrs
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        attrs: object = None,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        cached_device: Device = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        # deduce the device of the computation
        if cached_device is None:
            if not inputs:
                raise ValueError(
                    "Requires cached device to be available for tensor with no inputs"
                )
            cached_device = inputs[0].cached_device
            for x in inputs:
                if cached_device != x.cached_device:
                    raise ValueError(
                        "Requires all input devices to be the same to automatically"
                        "deduce device"
                    )
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.attrs = attrs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.cached_device = cached_device
        self.requires_grad = requires_grad

    @property
    def device(self):
        return self.cached_device

    @classmethod
    def make_const(cls, data, device, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            cached_device=device,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(
        cls, op: Op, inputs: List["Value"], *, attrs=None, cached_device=None
    ):
        value = cls.__new__(cls)
        value._init(op, inputs, attrs=attrs, cached_device=cached_device)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class Tuple(Value):
    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return simple_ml.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "simple_ml.Tuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, Tuple)
        assert len(self) == len(other)
        return simple_ml.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data(), self.device)


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self, array, *, device: Optional[Device] = None, dtype=None, requires_grad=True
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = device.array(array.numpy(), dtype=dtype)
        else:
            device = device if device else default_device()
            cached_data = device.array(array, dtype=dtype)

        self._init(
            None,
            [],
            cached_device=device,
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"], *, attrs=None, cached_device=None):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs, attrs=attrs, cached_device=cached_device)
        if not LAZY_MODE:
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, device, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            cached_device=device,
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.device == self.device and value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data(), self.device)

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def size(self):
        return self.realize_cached_data().size

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else simple_ml.ops.ones_like(self)
        compute_gradient_of_variables(self, out_grad)

    def __getitem__(self, idxs):
        return simple_ml.ops.get_item(self, idxs)

    def __setitem__(self, idxs, other):
        return simple_ml.ops.set_item(self, idxs, other)

    def __repr__(self):
        return "simple_ml.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        return self.device.to_numpy(self.realize_cached_data())

    def __add__(self, other):
        if isinstance(other, Tensor):
            return simple_ml.ops.add(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            return simple_ml.ops.add_scalar(self, other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return simple_ml.ops.multiply(self, other)
        else:
            return simple_ml.ops.multiply_scalar(self, other)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return simple_ml.ops.power_scalar(self, other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return simple_ml.ops.add(self, simple_ml.ops.negate(other))
        else:
            return simple_ml.ops.add_scalar(self, -other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return simple_ml.ops.divide(self, other)
        else:
            return simple_ml.ops.divide_scalar(self, other)

    def __matmul__(self, other):
        return simple_ml.ops.matmul(self, other)

    def matmul(self, other):
        return simple_ml.ops.matmul(self, other)

    def sum(self, axes=None):
        return simple_ml.ops.summation(self, axes)

    def broadcast_to(self, shape):
        return simple_ml.ops.broadcast_to(self, shape)

    def reshape(self, shape):
        return simple_ml.ops.reshape(self, shape)

    def __neg__(self):
        return simple_ml.ops.negate(self)

    def transpose(self, axes=None):
        return simple_ml.ops.transpose(self, axes)

    def flip(self, axes=None):
        return simple_ml.ops.flip(self, axes)

    def pad(self, axes=None):
        return simple_ml.ops.pad(self, axes)

    def permute(self, new_axes=None):
        return simple_ml.ops.permute(self, new_axes)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}  # defaultdict(list)
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        if node in node_to_output_grads_list:
            out_grad = sum_node_list(node_to_output_grads_list[node])
        else:
            out_grad = simple_ml.zeros_like(node)

        if not node.requires_grad:
            continue

        if not node.is_leaf():
            in_grads = node.op.gradient(out_grad, node)

        for i, x in enumerate(node.inputs):
            if x not in node_to_output_grads_list:
                node_to_output_grads_list[x] = []
            node_to_output_grads_list[x].append(in_grads[i])

        if node.is_leaf():
            node.grad = out_grad


class TraversalMark(Enum):
    NOT_VISITED = 0
    TEMPORARY = 1
    VISITED = 2


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """

    topo_order = []
    for node in node_list:
        if not hasattr(node, "mark"):
            topo_sort_dfs(node, topo_order)

    # clear marks on all nodes
    for node in topo_order:
        del node.mark

    # reverse post order dfs
    return topo_order[::-1]


def topo_sort_dfs(val: Value, topo_order: List[Value]):
    """Post-order DFS"""

    if not hasattr(val, "mark"):
        val.mark = TraversalMark.TEMPORARY
    elif val.mark == TraversalMark.VISITED:
        return
    elif val.mark == TraversalMark.TEMPORARY:
        raise ValueError("Not a DAG")

    for child in val.inputs:
        topo_sort_dfs(child, topo_order)

    val.mark = TraversalMark.VISITED
    topo_order.insert(0, val)


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from functools import reduce
    from operator import add

    return reduce(add, node_list)
