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
from op_test import OpTest, paddle_static_guard, skip_check_grad_ci

import paddle
from paddle import base


def group_norm_naive(x, scale, bias, epsilon, groups, data_layout):
    if data_layout == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
    N, C, H, W = x.shape
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    inv_var = 1.0 / np.sqrt(var + epsilon)
    output = (x - mean) * inv_var
    output = output.reshape((N, C, H, W)) * scale.reshape((-1, 1, 1)) + bias.reshape(
        (-1, 1, 1)
    )
    if data_layout == "NHWC":
        output = np.transpose(output, (0, 2, 3, 1))  # NCHW => NHWC
    return output, mean.reshape((N, G)), var.reshape((N, G))


class TestGroupNormOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):

            def test_x_type():
                input = np.random.random(2, 100, 3, 5).astype("float32")
                groups = 2
                paddle.static.nn.group_norm(input, groups)

            self.assertRaises(TypeError, test_x_type)

            def test_x_dtype():
                x2 = paddle.static.data(
                    name="x2", shape=[-1, 2, 100, 3, 5], dtype="int32"
                )
                groups = 2
                paddle.static.nn.group_norm(x2, groups)

            self.assertRaises(TypeError, test_x_dtype)
        paddle.disable_static()


def group_norm_wrapper(
    input, weight, bias, epsilon=1e-5, num_groups=0, data_format="NCHW"
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle._C_ops.group_norm(
        input, weight, bias, epsilon, num_groups, data_format
    )


@skip_check_grad_ci("group_norm_grad not support on sdaa")
class TestGroupNormOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "group_norm"
        self.python_api = group_norm_wrapper
        self.python_out_sig = ["Y"]
        self.data_format = "NHWC"
        self.atol = 1e-6
        self.dtype = np.float32
        self.shape = (2, 100, 3, 5)
        self.attrs = {"epsilon": 1e-5, "groups": 2, "data_layout": "NCHW"}
        self.compare_between_place = False
        self.init_test_case()

        input = np.random.random(self.shape).astype(self.dtype)
        if self.data_format == "NHWC":
            input = np.transpose(input, (0, 2, 3, 1))
        scale = np.random.random([self.shape[1]]).astype(self.dtype)
        bias = np.random.random([self.shape[1]]).astype(self.dtype)
        output, mean, var = group_norm_naive(
            input,
            scale,
            bias,
            self.attrs["epsilon"],
            self.attrs["groups"],
            self.data_format,
        )

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(input),
            "Scale": OpTest.np_dtype_to_base_dtype(scale),
            "Bias": OpTest.np_dtype_to_base_dtype(bias),
        }
        self.outputs = {"Y": output, "Mean": mean, "Variance": var}
        self.attrs["data_layout"] = self.data_format

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=self.atol)

    def init_test_case(self):
        pass


class TestGroupNormFP16OP(TestGroupNormOp):
    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_test_case(self):
        self.dtype = np.float16


class TestGroupNormOp1(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 1


class TestGroupNormFP16Op1(TestGroupNormFP16OP):
    def init_test_case(self):
        self.attrs["groups"] = 1
        self.dtype = np.float16


class TestGroupNormOp2(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 4


class TestGroupNormFP16Op2(TestGroupNormFP16OP):
    def init_test_case(self):
        self.attrs["groups"] = 4
        self.dtype = np.float16


class TestGroupNormOpBigEps1(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 1
        self.attrs["epsilon"] = 0.5


class TestGroupNormOpBigEps2(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 4
        self.attrs["epsilon"] = 0.5


class TestGroupNormOpBigEps3(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["epsilon"] = 0.5


class TestGroupNormOp1_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 1
        self.data_format = "NHWC"


class TestGroupNormOp2_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 4
        self.data_format = "NHWC"


class TestGroupNormFP16Op_With_NHWC(TestGroupNormFP16OP):
    def init_test_case(self):
        self.no_need_check_inplace = True
        self.attrs["groups"] = 1
        self.data_format = "NHWC"
        self.attrs["epsilon"] = 0.5
        self.shape = (1, 100, 4, 4)
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestGroupNormOpBigEps1_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 1
        self.attrs["epsilon"] = 0.5
        self.data_format = "NHWC"


class TestGroupNormOpBigEps2_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["groups"] = 4
        self.attrs["epsilon"] = 0.5
        self.data_format = "NHWC"


class TestGroupNormOpBigEps3_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs["epsilon"] = 0.5
        self.data_format = "NHWC"


class TestGroupNormAPI_With_NHWC(unittest.TestCase):
    def test_case1(self):
        with paddle.pir_utils.OldIrGuard():
            with paddle_static_guard():
                data1 = paddle.static.data(
                    name="data1", shape=[None, 3, 3, 4], dtype="float32"
                )
                out1 = paddle.static.nn.group_norm(
                    input=data1, groups=2, data_layout="NHWC"
                )
                data2 = paddle.static.data(
                    name="data2", shape=[None, 4, 3, 3], dtype="float32"
                )
                out2 = paddle.static.nn.group_norm(
                    input=data2, groups=2, data_layout="NCHW"
                )

                data1_np = np.random.random((2, 3, 3, 4)).astype("float32")
                data2_np = np.random.random((2, 4, 3, 3)).astype("float32")
                scale = np.array([1]).astype("float32")
                bias = np.array([0]).astype("float32")

                place = paddle.CustomPlace("sdaa", 0)
                exe = base.Executor(place)
                results = exe.run(
                    base.default_main_program(),
                    feed={"data1": data1_np, "data2": data2_np},
                    fetch_list=[out1, out2],
                    return_numpy=True,
                )
                expect_res1 = group_norm_naive(
                    data1_np,
                    scale,
                    bias,
                    epsilon=1e-5,
                    groups=2,
                    data_layout="NHWC",
                )
                expect_res2 = group_norm_naive(
                    data2_np,
                    scale,
                    bias,
                    epsilon=1e-5,
                    groups=2,
                    data_layout="NCHW",
                )
                np.testing.assert_allclose(
                    results[0], expect_res1[0], rtol=1e-05, atol=1e-6
                )
                np.testing.assert_allclose(
                    results[1], expect_res2[0], rtol=1e-05, atol=1e-6
                )


class TestGroupNormException(unittest.TestCase):
    # data_layout is not NHWC or NCHW
    def test_exception(self):
        paddle.enable_static()
        data = paddle.static.data(name="data", shape=[None, 3, 3, 4], dtype="float32")

        def attr_data_format():
            out = paddle.static.nn.group_norm(input=data, groups=2, data_layout="NDHW")

        self.assertRaises(ValueError, attr_data_format)

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
