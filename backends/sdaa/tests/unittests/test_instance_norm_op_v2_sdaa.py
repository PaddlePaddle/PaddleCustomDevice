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
import numpy as np
import paddle
from paddle import base
from op_test import OpTest, paddle_static_guard

paddle.enable_static()
SEED = 2021


def instance_norm_wrapper(
    input, weight, bias, epsilon=1e-5, momentum=0.9, data_format="NCHW"
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle.nn.functional.instance_norm(
        input, None, None, weight, bias, True, momentum, epsilon, data_format
    )


def _reference_instance_norm(x, scale, bias, epsilon):
    N, C, H, W = x.shape
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    variance = np.var(x, axis=(2, 3), keepdims=True)
    std = np.sqrt(variance) + epsilon
    x_norm = (x - mean) / std
    scale = scale.reshape([1, C, 1, 1])
    bias = bias.reshape([1, C, 1, 1])
    x_norm = scale * x_norm + bias
    return x_norm, mean.reshape(N * C), variance.reshape(N * C)


class TestInstanceNorm(unittest.TestCase):
    def test_error(self):
        places = [paddle.CustomPlace("sdaa", 0)]

        for p in places:

            def error1d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype("float32")
                instance_norm1d = paddle.nn.InstanceNorm1D(1)
                instance_norm1d(paddle.to_tensor(x_data_4))

            def error2d():
                x_data_3 = np.random.random(size=(2, 1, 3)).astype("float32")
                instance_norm2d = paddle.nn.InstanceNorm2D(1)
                instance_norm2d(paddle.to_tensor(x_data_3))

            def error3d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype("float32")
                instance_norm3d = paddle.nn.InstanceNorm3D(1)
                instance_norm3d(paddle.to_tensor(x_data_4))

            def weight_bias_false():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype("float32")
                instance_norm3d = paddle.nn.InstanceNorm3D(
                    1, weight_attr=False, bias_attr=False
                )

            with base.dygraph.guard(p):
                weight_bias_false()
                self.assertRaises(ValueError, error1d)
                self.assertRaises(ValueError, error2d)
                self.assertRaises(ValueError, error3d)

    def test_dygraph(self):
        places = [paddle.CustomPlace("sdaa", 0)]
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with base.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(paddle.to_tensor(x))
                return y.numpy()

            def compute_v2(x):
                with base.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(paddle.to_tensor(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


class TestInstanceNormFP32OP(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        """Test instance_norm op with default value"""
        self.set_sdaa()
        self.op_type = "instance_norm"
        self.__class__.op_type = self.op_type
        self.data_format = "NCHW"
        self.eps = 1e-5
        self.init_dtype()
        self.init_shape()
        self.init_value()
        self.inputs = {"X": self.value, "Scale": self.scale, "Bias": self.bias}
        self.attrs = {
            "epsilon": self.eps,
            "momentum": 0.9,
            "data_format": self.data_format,
        }
        y, mean, variance = _reference_instance_norm(
            self.value, self.scale, self.bias, self.eps
        )

        self.python_out_sig = ["Y"]
        self.outputs = {
            "Y": y,
            "SavedMean": mean,
            "SavedVariance": variance,
        }
        self.python_api = instance_norm_wrapper
        self.public_python_api = instance_norm_wrapper

        if self.dtype == np.float16:
            self.numeric_place = None
        else:
            self.numeric_place = paddle.CPUPlace()

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.003)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X", "Scale", "Bias"], "Y", numeric_place=self.numeric_place
        )

    def test_check_grad_no_scale_bias(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Y",
            no_grad_set=set(["Scale", "Bias"]),
            numeric_place=self.numeric_place,
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [4, 100, 4, 4]

    def init_value(self):
        np.random.seed(0)
        self.value = np.random.random(self.shape).astype(self.dtype)
        self.scale = np.random.random([self.shape[1]]).astype(np.float32)
        self.bias = np.random.random([self.shape[1]]).astype(np.float32)


class TestInstanceNormFP16OP(TestInstanceNormFP32OP):
    def setUp(self):
        super().setUp()

    def init_dtype(self):
        self.dtype = np.float16


class TestInstanceNormFP32OPwithoutScaleBias(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        """Test instance_norm op with default value"""
        self.set_sdaa()
        self.op_type = "instance_norm"
        self.__class__.op_type = self.op_type
        self.data_format = "NCHW"
        self.eps = 1e-5
        self.init_dtype()
        self.init_shape()
        self.init_value()
        self.inputs = {"X": self.value}
        self.attrs = {
            "epsilon": self.eps,
            "momentum": 0.9,
            "data_format": self.data_format,
        }
        y, mean, variance = _reference_instance_norm(
            self.value, self.scale, self.bias, self.eps
        )

        self.python_out_sig = ["Y"]
        self.outputs = {
            "Y": y,
            "SavedMean": mean,
            "SavedVariance": variance,
        }
        self.python_api = instance_norm_wrapper
        self.public_python_api = instance_norm_wrapper

        if self.dtype == np.float16:
            self.numeric_place = None
        else:
            self.numeric_place = paddle.CPUPlace()

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.003)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Y", numeric_place=self.numeric_place
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [4, 100, 4, 4]

    def init_value(self):
        np.random.seed(0)
        self.value = np.random.random(self.shape).astype(self.dtype)
        self.scale = np.ones([self.shape[1]]).astype(np.float32)
        self.bias = np.zeros([self.shape[1]]).astype(np.float32)


class TestInstanceNormFP16OPwithoutScaleBias(TestInstanceNormFP32OPwithoutScaleBias):
    def setUp(self):
        super().setUp()

    def init_dtype(self):
        self.dtype = np.float16


class TestInstanceNormAPI(unittest.TestCase):
    def test_case1(self):
        with paddle.pir_utils.OldIrGuard():
            with paddle_static_guard():
                data1 = paddle.static.data(
                    name="data1", shape=[None, 3, 3, 4], dtype="float32"
                )
                out1 = paddle.static.nn.instance_norm(input=data1)
                data2 = paddle.static.data(
                    name="data2", shape=[None, 4, 3, 3], dtype="float32"
                )
                out2 = paddle.static.nn.instance_norm(input=data2)

                shape1_np = (2, 3, 3, 4)
                shape2_np = (2, 4, 3, 3)

                data1_np = np.random.random(shape1_np).astype("float32")
                scale1_np = np.ones([shape1_np[1]]).astype(np.float32)
                bias1_np = np.zeros([shape1_np[1]]).astype(np.float32)

                data2_np = np.random.random(shape2_np).astype("float32")
                scale2_np = np.ones([shape2_np[1]]).astype(np.float32)
                bias2_np = np.zeros([shape2_np[1]]).astype(np.float32)

                place = paddle.CustomPlace("sdaa", 0)
                exe = base.Executor(place)
                results = exe.run(
                    base.default_main_program(),
                    feed={"data1": data1_np, "data2": data2_np},
                    fetch_list=[out1, out2],
                    return_numpy=True,
                )
                expect_res1 = _reference_instance_norm(
                    data1_np,
                    scale1_np,
                    bias1_np,
                    epsilon=1e-5,
                )
                expect_res2 = _reference_instance_norm(
                    data2_np,
                    scale2_np,
                    bias2_np,
                    epsilon=1e-5,
                )
                np.testing.assert_allclose(
                    results[0], expect_res1[0], rtol=1e-05, atol=1e-03
                )
                np.testing.assert_allclose(
                    results[1], expect_res2[0], rtol=1e-05, atol=1e-03
                )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
