import math
import operator
from functools import reduce

import numpy as np

from . import ndarray_backend_cpu, ndarray_backend_numpy


# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def opencl():
    """Return opencl device"""
    try:
        from . import ndarray_backend_opencl

        return BackendDevice("opencl", ndarray_backend_opencl)
    except ImportError:
        return BackendDevice("opencl", None)


def numpy_device():
    """Return numpy device"""
    return BackendDevice("numpy_device", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return numpy_device()


class NDArray:
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle
        self._view = False

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        array._view = False
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
            array._view = True
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def view(self):
        return self._view

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    def astype(self, dtype):
        # only support float32 for now
        return self

    ### Basic array manipulation
    def fill(self, value):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """
        # TODO: a reshape on a view (based on get item) should sometimes
        #       trigger a copy because reshape on a view does not always work
        #       I had to implement view based auto-compacting, I need to extend
        #       this functionality else where you can end up operating on a view
        #       https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

        if self.view:
            self._init(self.compact())

        if new_shape == tuple():
            curr_dim = reduce(operator.mul, self.shape)
            new_strides = tuple()
            if curr_dim != 1:
                raise ValueError("cannot reshape to scalar when shape is not all ones")
        else:
            new_dim = reduce(operator.mul, new_shape)
            if len(self.shape) == 0:
                if new_dim != 1:
                    raise ValueError(
                        "cannot reshape from scalar to an array with any "
                        "dimensions greater than one"
                    )
                # set curr_dim to one to set up for the curr_dim != new_dim test
                curr_dim = 1
            else:
                curr_dim = reduce(operator.mul, self.shape)
            if curr_dim != new_dim:
                raise ValueError(
                    f"cannot reshape array of shape {self.shape} into " f"{new_shape}"
                )
            _dim = new_dim
            new_strides = []
            for d in new_shape:
                _dim //= d
                new_strides.append(_dim)

        return self.as_strided(new_shape, tuple(new_strides))

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        if len(new_axes) != len(self.shape):
            raise ValueError("new_axes dimensions does not match input axes dimensions")
        if set(new_axes) != set(range(len(self.shape))):
            raise ValueError("new_axes does not contain all axes")

        new_shape = tuple(map(self.shape.__getitem__, new_axes))
        new_strides = tuple(map(self.strides.__getitem__, new_axes))
        return self.as_strided(new_shape, new_strides)

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        if not all(e1 == e2 for e1, e2 in zip(self.shape, new_shape) if e1 != 1):
            raise ValueError(
                f"cannot broadcast array of shape {self.shape} into {new_shape}"
            )

        new_strides = tuple(
            stride if shape != 1 else 0
            for stride, shape in zip(self.strides, self.shape)
        )
        return self.as_strided(new_shape, new_strides)

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        proc_idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(proc_idxs) == self.ndim, "Need indexes equal to number of dimensions"

        # calculate shape
        shape = tuple(
            map(
                # get_slice_len function
                lambda idx: math.ceil((idx.stop - idx.start) / idx.step),
                proc_idxs,
            )
        )

        # calculate offset
        idx_start = [idx.start for idx in proc_idxs]
        idx_stride_prod = map(operator.mul, self.strides, idx_start)
        offset = reduce(operator.add, idx_stride_prod)

        # calculate strides
        strides = tuple(
            map(operator.mul, self.strides, [idx.step for idx in proc_idxs])
        )

        # Here, we reduce the dimension for any dimension in the original idx
        # slices that were ints and not slices.
        # Example:
        #     X[3:4, 3:4] = [[z]]
        #     X[3, 3] = z
        if any(not isinstance(s, slice) for s in idxs):
            _shape = tuple(
                shape_i for idx, shape_i in zip(idxs, shape) if isinstance(idx, slice)
            )
            _strides = tuple(
                stride for idx, stride in zip(idxs, strides) if isinstance(idx, slice)
            )
            shape = _shape
            strides = _strides

        return NDArray.make(
            shape=shape,
            strides=strides,
            offset=offset,
            device=self.device,
            handle=self._handle,
        )

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Convolution
    def conv2(self, other):
        assert self.ndim == 2 and other.ndim == 2
        assert other.shape[0] == other.shape[1]
        assert other.shape[0] <= self.shape[0]
        assert other.shape[0] <= self.shape[1]

        m, n, k = self.shape[0], self.shape[1], other.shape[0]

        if hasattr(self.device, "conv2"):
            out = NDArray.make((m - k + 1, n - k + 1), device=self.device)
            self.device.conv2(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, k
            )
            return out
        else:
            raise ValueError("Convolution is not implemented on this device.")

    ### Convolution
    def conv4(self, other, stride=1):
        # self  : n h w    c_in
        # other : k k c_in c_out
        assert self.ndim == 4 and other.ndim == 4
        assert other.shape[0] == other.shape[1]
        assert self.shape[-1] == other.shape[2]

        n, h, w, c_in, c_out, k = (
            *self.shape,
            other.shape[-1],
            other.shape[0],
        )

        if hasattr(self.device, "conv4"):
            out = NDArray.make(
                (n, (h - k) // stride + 1, (w - k) // stride + 1, c_out),
                device=self.device,
            )
            self.device.conv4(
                self.compact()._handle,
                other.compact()._handle,
                out._handle,
                n,
                h,
                w,
                c_in,
                c_out,
                k,
                stride,
            )
            return out
        else:
            raise ValueError("Convolution is not implemented on this device.")

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis):
        """Return a view to the array set up for reduction functions and output array."""
        if axis is None:
            view = self.reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,) * self.ndim, device=self.device)
        else:
            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)]),
                device=self.device,
            )
        return view, out

    def sum(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None):
        view, out = self.reduce_view_out(axis)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.

        Note: compact() before returning.
        """

        if axes is None:
            axes = range(len(self.shape))
        offset = 0
        acc_prod = 1
        strides = list(self.strides)
        for i, r in reversed(list(enumerate(self.shape))):
            acc_prod *= r
            if i in axes:
                strides[i] *= -1
                offset += strides[i] + acc_prod
        return NDArray.make(
            self.shape,
            strides=strides,
            device=self.device,
            handle=self._handle,
            offset=offset,
        ).compact()

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """

        shape_mods = map(sum, axes)
        new_shape = tuple(map(sum, zip(self.shape, shape_mods)))
        out = empty(new_shape, device=self.device)
        out.fill(0.0)
        slices = [
            slice(None) if lp == rp == 0 else slice(lp, lp + dim)
            for dim, (lp, rp) in zip(self.shape, axes)
        ]
        out[tuple(slices)] = self[tuple(slice(None) for _ in range(len(self.shape)))]
        return out


def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray.make(shape, device=device)
