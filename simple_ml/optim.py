"""Optimization module"""
import numpy as np

import simple_ml as sm


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.delta = {}
        self.weight_decay = weight_decay

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(
            np.array(
                [
                    np.linalg.norm(p.grad.detach().numpy()).reshape((1,))
                    for p in self.params
                ]
            )
        )
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped

    def step(self):

        for i, param in enumerate(self.params):
            grad = param.grad.data + self.weight_decay * param.data
            if i not in self.delta:
                self.delta[i] = sm.Tensor.make_const(
                    -self.lr * grad, device=param.device
                )
            else:
                self.delta[i].data = (
                    self.momentum * self.delta[i].data + -self.lr * grad
                )
            param.data = param.data + self.delta[i].data


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        bias_correction=True,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):

        self.t += 1
        for i, param in enumerate(self.params):
            grad = param.grad.data + self.weight_decay * param.data
            if i not in self.m:
                self.m[i] = sm.Tensor.make_const(
                    (1 - self.beta1) * grad, device=param.device
                )
            else:
                self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad

            if i not in self.v:
                self.v[i] = sm.Tensor.make_const(
                    (1 - self.beta2) * grad**2, device=param.device
                )
            else:
                self.v[i].data = (
                    self.beta2 * self.v[i].data + (1 - self.beta2) * grad**2
                )

            m_hat = self.m[i].data
            v_hat = self.v[i].data

            if self.bias_correction:
                m_hat = m_hat / (1 - self.beta1**self.t)
                v_hat = v_hat / (1 - self.beta2**self.t)

            param.data = param.data + -self.lr * m_hat / (v_hat**0.5 + self.eps)

            del m_hat, v_hat
