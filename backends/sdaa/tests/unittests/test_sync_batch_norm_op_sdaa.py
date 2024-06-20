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

import os
import unittest
import paddle
import numpy as np

from test_dist_base import TestDistBase

flag_name = os.path.splitext(__file__)[0]


def _reference_testing(x, scale, offset, mean, var, epsilon, data_format):
    x_shape = x.shape
    if len(x_shape) == 2:
        if data_format == "NCHW":
            x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        else:
            x = np.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))

    if data_format == "NCHW":
        n, c, h, w = x.shape
        mean_tile = np.reshape(mean, (1, c, 1, 1))
        mean_tile = np.tile(mean_tile, (n, 1, h, w))
        var_tile = np.reshape(var, (1, c, 1, 1))
        var_tile = np.tile(var_tile, (n, 1, h, w))
        normalized = (x - mean_tile) / np.sqrt(var_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        offset_tile = np.reshape(offset, (1, c, 1, 1))
        offset_tile = np.reshape(offset_tile, (1, c, 1, 1))
        y = normalized * scale_tile + offset_tile
    elif data_format == "NHWC":
        normalized = (x - mean) / np.sqrt(var + epsilon)
        y = normalized * scale + offset
    else:
        raise ValueError("Unknown data order.")

    if len(x_shape) == 2 or len(x_shape) == 3:
        y = np.reshape(y, x_shape)
    return y


class TestParallelDygraphMnist(TestDistBase):
    def _setup_config(self):
        self._dygraph = True
        self._use_sdaa = True
        self._use_NHWC = False
        self._ignore_syncbn_bias_grad = False

    def test_mnist(self):
        self.check_with_place(
            "parallel_dygraph_sync_batch_norm.py",
            delta=1e-5,
        )


class TestParallelDygraphMnist_channel_last(TestParallelDygraphMnist):
    def _setup_config(self):
        self._dygraph = True
        self._use_sdaa = True
        self._use_NHWC = True
        self._ignore_syncbn_bias_grad = False


class TestParallelDygraphMnist_skip_bias_scale_grad(TestParallelDygraphMnist):
    def _setup_config(self):
        self._dygraph = True
        self._use_sdaa = True
        self._use_NHWC = True
        self._ignore_syncbn_bias_grad = True


class TestSyncBatchNormOpInference(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.data_formats = ["NCHW", "NHWC"]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def check_with_place(self, data_layout, dtype, shape):
        paddle.device.set_device("sdaa")
        epsilon = 0.00001
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        if data_layout == "NHWC":
            x_shape = [n, h, w, c]
        elif data_layout == "NCHW":
            x_shape = [n, c, h, w]
        else:
            raise ValueError("Unknown data layout.")
        scale_shape = [c]

        x = np.random.random_sample(x_shape).astype(dtype)
        x = x - 0.5
        scale = np.random.random_sample(scale_shape).astype(np.float32)
        bias = np.random.random_sample(scale_shape).astype(np.float32)
        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.ones(scale_shape).astype(np.float32)
        y = _reference_testing(
            x, scale, bias, mean, variance, epsilon, data_layout
        ).astype(dtype)

        sync_batch_norm_out, _, _, _, _, _ = paddle._C_ops.sync_batch_norm_(
            paddle.to_tensor(x),
            paddle.to_tensor(mean),
            paddle.to_tensor(variance),
            paddle.to_tensor(scale),
            paddle.to_tensor(bias),
            True,
            0.9,
            epsilon,
            data_layout,
            False,
            False,
        )

        self.__assert_close(sync_batch_norm_out, y, "y", atol=1e-3)

    def test_check_output(self):
        paddle.disable_static()
        for data_format in self.data_formats:
            self.check_with_place(data_format, self.dtype, [2, 3, 4, 5])

        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
