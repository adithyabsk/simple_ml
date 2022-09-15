import sys

sys.path.append("./python")
import itertools

import numpy as np
import pytest
import torch

import simple_ml as sm
from simple_ml import backend_ndarray as nd

np.random.seed(1)


def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = f.gradient(sm.Tensor(c, device=args[i].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1
    return np.array([g.numpy() for g in backward_grad])


_DEVICES = [
    sm.cpu(),
    pytest.param(
        sm.cuda(), marks=pytest.mark.skipif(not sm.cuda().enabled(), reason="No GPU")
    ),
]


EWISE_OPS = {"divide": lambda a, b: a / b, "subtract": lambda a, b: a - b}
EWISE_OP_FNS = [EWISE_OPS[k] for k in EWISE_OPS]
EWISE_OP_NAMES = [k for k in EWISE_OPS]
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]


@pytest.mark.parametrize("fn", EWISE_OP_FNS, ids=EWISE_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    B = sm.Tensor(nd.array(_B), device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)


SCALAR_OPS = {"divide": lambda a, b: a / b, "subtract": lambda a, b: a - b}
SCALAR_OP_FNS = [SCALAR_OPS[k] for k in SCALAR_OPS]
SCALAR_OP_NAMES = [k for k in SCALAR_OPS]


@pytest.mark.parametrize("fn", SCALAR_OP_FNS, ids=SCALAR_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(1).astype(np.float32).item()
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, _B).numpy(), atol=1e-5, rtol=1e-5)


MATMUL_DIMS = [
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 32),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]


@pytest.mark.parametrize("m,n,p", MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    B = sm.Tensor(nd.array(_B), device=device)
    np.testing.assert_allclose(_A @ _B, (A @ B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_power(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randint(1)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(_A**_B, (A**_B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_log(shape, device):
    _A = np.random.randn(*shape).astype(np.float32) + 5.0
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.log(_A), sm.log(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_exp(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.exp(_A), sm.exp(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_relu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.maximum(_A, 0), sm.relu(A).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_tanh(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.tanh(_A), sm.tanh(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_tanh_backward(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    backward_check(sm.tanh, A)


STACK_PARAMETERS = [((5, 5), 0, 1), ((1, 5, 7), 2, 5)]


@pytest.mark.parametrize("shape, axis, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [sm.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    out = sm.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)
    np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape, axis, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack_backward(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [sm.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    for i in range(l):
        A_t[i].requires_grad = True
    sm.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()
    for i in range(l):
        np.testing.assert_allclose(
            A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5
        )


SUMMATION_PARAMETERS = [
    ((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), (1, 2)),
]


@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.sum(_A, axes), sm.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation_backward(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    backward_check(sm.summation, A, axes=axes)


BROADCAST_SHAPES = [((1, 1, 1), (3, 3, 3)), ((4, 1, 6), (4, 3, 6))]


@pytest.mark.parametrize("shape,shape_to", BROADCAST_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.broadcast_to(_A, shape_to),
        sm.broadcast_to(A, shape_to).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


RESHAPE_SHAPES = [((1, 1, 1), (1,)), ((4, 1, 6), (6, 4, 1))]


@pytest.mark.parametrize("shape,shape_to", RESHAPE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.reshape(_A, shape_to), sm.reshape(A, shape_to).numpy(), atol=1e-5, rtol=1e-5
    )


TRANSPOSE_SHAPES = [(1, 1, 1), (4, 5, 6)]
TRANSPOSE_AXES = [(0, 1), (0, 2), None]


@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("axes", TRANSPOSE_AXES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_transpose(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    if axes is None:
        np_axes = (_A.ndim - 2, _A.ndim - 1)
    else:
        np_axes = axes
    np.testing.assert_allclose(
        np.swapaxes(_A, np_axes[0], np_axes[1]),
        sm.transpose(A, axes=axes).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


PERMUTE_SHAPES = [(1, 1, 1), (4, 5, 6)]
PERMUTE_AXES = [(1, 0, 2), (0, 2, 1)]


@pytest.mark.parametrize("shape", PERMUTE_SHAPES)
@pytest.mark.parametrize("axes", PERMUTE_AXES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_permute(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.transpose(_A, axes=axes),
        sm.permute(A, new_axes=axes).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", PERMUTE_SHAPES)
@pytest.mark.parametrize("axes", PERMUTE_AXES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_permute_backward(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    backward_check(sm.permute, A, new_axes=axes)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsoftmax(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    A_t = torch.Tensor(_A)
    np.testing.assert_allclose(
        torch.log_softmax(A_t, axis=-1).numpy(),
        sm.logsoftmax(A).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


GETSETITEM_PARAMS = [
    ((1, 1), (0, 0)),
    ((4, 10), (np.s_[1:3], np.s_[5:10])),
    ((4, 5, 10), (1, np.s_[1:], np.s_[:5])),
    ((4, 1, 1), (1, np.s_[:], np.s_[:])),
    ((1, 4, 1), (np.s_[:], 1, np.s_[:])),
]


@pytest.mark.parametrize("shape,idxs", GETSETITEM_PARAMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem(shape, idxs, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(_A[idxs], A[idxs].numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape,idxs", GETSETITEM_PARAMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem_backward(shape, idxs, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    backward_check(sm.get_item, A, idxs=idxs)


@pytest.mark.parametrize("shape,idxs", GETSETITEM_PARAMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem(shape, idxs, device):
    _A = np.random.randn(*shape).astype(np.float32)
    shape = _A[idxs].shape
    if len(shape) == 0:
        shape = (1,)
    _B = np.random.randn(*shape).astype(np.float32)
    A = sm.Tensor(nd.array(_A), device=device)
    B = sm.Tensor(nd.array(_B), device=device)
    A[idxs] = B
    _A[idxs] = _B
    np.testing.assert_allclose(_A, A.numpy(), atol=1e-5, rtol=1e-5)
