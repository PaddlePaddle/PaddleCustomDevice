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

from paddle.nn import Conv2D, BatchNorm

import numpy as np

from paddle.quantization.imperative.fuse_utils import _fuse_conv_bn_eval
from paddle import ParamAttr

SEED = 2023

np.random.seed(SEED)
paddle.seed(SEED)


class custom_op_test(unittest.TestCase):
    def test_custom_add(self):
        paddle.set_device("sdaa")
        x = paddle.rand(shape=[2, 3], dtype="float32")
        y = paddle.rand(shape=[2, 3], dtype="float32")
        p_results = paddle.add(x, y)
        c_results = paddle_sdaa.custom_add(x, y)
        self.assertTrue(np.allclose(p_results, c_results))

    def test_custom_add_n(self):
        paddle.set_device("sdaa")
        x = paddle.rand(shape=[2, 3], dtype="float32")
        y = paddle.rand(shape=[2, 3], dtype="float32")
        z = paddle.rand(shape=[2, 3], dtype="float32")
        p_results = paddle.add(x, y)
        p_results = paddle.add(p_results, z)
        c_results = paddle_sdaa.custom_add_n(x, y, z)
        self.assertTrue(np.allclose(p_results, c_results))

    def test_custom_fc(self):
        batch_size = 64
        input = np.random.rand(batch_size, 200).astype("float32")
        bias = np.random.rand(2).astype("float32")
        w = np.random.rand(200, 2).astype("float32")

        with paddle.base.dygraph.guard(place=paddle.CPUPlace()):
            cpu_output = paddle._legacy_C_ops.fc(
                paddle.to_tensor(input),
                paddle.to_tensor(w),
                paddle.to_tensor(bias),
                "activation_type",
                "",
                "in_num_col_dims",
                1,
            )

        paddle.set_device("sdaa")
        sdaa_output = paddle_sdaa.custom_fc(
            input=paddle.to_tensor(input),
            w=paddle.to_tensor(w),
            bias=paddle.to_tensor(bias),
            in_num_col_dims=1,
            activation_type="",
            padding_weights=False,
        )

        self.assertTrue(np.allclose(sdaa_output.numpy(), cpu_output.numpy()))


class TestCustomConvBN(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_inputs()

    def init_inputs(self):
        self.input = np.random.randn(8, 384, 20, 20).astype("float32")

    def init_params(self):
        self.num_channels = 384
        self.num_filters = 384
        self.filter_size = 3
        self.stride = 1
        self.groups = 1
        self.data_format = "NCHW"

    def test_custom_conv_bn(self):

        paddle.set_device("cpu")
        conv = Conv2D(
            in_channels=self.num_channels,
            out_channels=self.num_filters,
            kernel_size=self.filter_size,
            stride=self.stride,
            padding=(self.filter_size - 1) // 2,
            groups=self.groups,
            weight_attr=ParamAttr(
                learning_rate=0.1, initializer=paddle.nn.initializer.KaimingNormal()
            ),
            bias_attr=False,
            data_format=self.data_format,
        )
        conv.training = False

        conv_filter = conv.weight.clone()

        bn = BatchNorm(
            self.num_filters,
            param_attr=ParamAttr(learning_rate=0.1),
            bias_attr=ParamAttr(learning_rate=0.1),
            data_layout=self.data_format,
        )
        bn.training = False

        bn_scale = bn.weight.clone()
        bn_bias = bn.bias.clone()
        bn_mean = bn._mean.clone()
        bn_var = bn._variance.clone()

        conv_bn = _fuse_conv_bn_eval(conv, bn)
        cpu_input = paddle.to_tensor(self.input, place=paddle.CPUPlace())
        cpu_output = conv_bn(cpu_input)

        paddle.set_device("sdaa")
        sdaa_input = paddle.to_tensor(self.input)
        filter = paddle.to_tensor(conv_filter.numpy())
        scale = paddle.to_tensor(bn_scale.numpy())
        bias = paddle.to_tensor(bn_bias.numpy())
        mean = paddle.to_tensor(bn_mean.numpy())
        var = paddle.to_tensor(bn_var.numpy())

        sdaa_output = paddle_sdaa.custom_fused_conv_bn(
            input=sdaa_input,
            filter=filter,
            scale=scale,
            bias=bias,
            mean=mean,
            var=var,
            strides=conv._stride,
            paddings=conv._updated_padding,
            padding_algorithm=conv._padding_algorithm,
            dilations=conv._dilation,
            groups=conv._groups,
            data_format=conv._data_format,
            epsilon=bn._epsilon,
            activation_type="",
        )

        np.testing.assert_allclose(sdaa_output.numpy(), cpu_output.numpy(), atol=1e-2)


if __name__ == "__main__":
    unittest.main()
