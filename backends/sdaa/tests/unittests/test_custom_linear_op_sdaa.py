# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import unittest
import paddle
import paddle_sdaa

import numpy as np

SEED = 2023

np.random.seed(SEED)
paddle.seed(SEED)


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float64")
    return Out


class TestCustomLinear(unittest.TestCase):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_testcase(self):
        self.x = np.random.random([4096, 3840]).astype(np.float32)
        self.weight = np.random.random([3840, 5120]).astype(np.float32)
        self.bias = np.random.random([4096, 5120]).astype(np.float32)
        self.x = -0.1 + 0.2 * self.x
        self.weight = -0.1 + 0.2 * self.weight

    def test_linear(self):
        self.init_testcase()
        self.set_sdaa()
        x_gt = np.copy(self.x)
        weight_gt = np.copy(self.weight)
        bias_gt = np.copy(self.bias)

        x_custom = paddle.to_tensor(self.x, stop_gradient=False)
        weight_custom = paddle.to_tensor(self.weight, stop_gradient=False)
        bias_custom = paddle.to_tensor(self.bias, stop_gradient=False)

        out_custom = paddle_sdaa.ops.linear(x_custom, weight_custom, bias_custom)
        out_custom.backward()

        out_gt = reference_matmul(x_gt, weight_gt) + bias_gt
        x_gt_grad = reference_matmul(np.ones_like(out_gt), weight_gt.T)
        weight_gt_grad = reference_matmul(x_gt.T, np.ones_like(out_gt))

        np.testing.assert_allclose(out_custom.numpy(), out_gt, atol=0.001)

        np.testing.assert_allclose(x_custom.grad.numpy(), x_gt_grad, rtol=1.5)

        np.testing.assert_allclose(weight_custom.grad.numpy(), weight_gt_grad, rtol=1.5)


class TestCustomLinearWithoutBias(unittest.TestCase):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_testcase(self):
        self.x = np.random.random([4096, 3840]).astype(np.float32)
        self.weight = np.random.random([3840, 5120]).astype(np.float32)
        self.x = -0.1 + 0.2 * self.x
        self.weight = -0.1 + 0.2 * self.weight

    def test_linear(self):
        self.init_testcase()
        self.set_sdaa()
        x_gt = np.copy(self.x)
        weight_gt = np.copy(self.weight)

        x_custom = paddle.to_tensor(self.x, stop_gradient=False)
        weight_custom = paddle.to_tensor(self.weight, stop_gradient=False)

        out_custom = paddle_sdaa.ops.linear(x_custom, weight_custom)
        out_custom.backward()

        out_gt = reference_matmul(x_gt, weight_gt)
        x_gt_grad = reference_matmul(np.ones_like(out_gt), weight_gt.T)
        weight_gt_grad = reference_matmul(x_gt.T, np.ones_like(out_gt))

        np.testing.assert_allclose(out_custom.numpy(), out_gt, atol=0.001)

        np.testing.assert_allclose(x_custom.grad.numpy(), x_gt_grad, rtol=1.5)

        np.testing.assert_allclose(weight_custom.grad.numpy(), weight_gt_grad, rtol=1.5)


if __name__ == "__main__":
    unittest.main()
