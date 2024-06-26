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
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import Program


def ref_prelu(x, weight):
    x_t = x.copy()
    weight = weight.reshape(1, -1, 1, 1)
    neg_indices = x <= 0
    assert x.shape == neg_indices.shape
    x_t[neg_indices] = (x_t * weight)[neg_indices]
    return x_t


def ref_prelu_nn(x, num_parameters, init):
    weight_np = np.full((num_parameters), init)
    return ref_prelu(x, weight_np)


class TestFunctionalPReluAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.x_np = np.random.uniform(-1.0, 1.0, [1, 2, 3, 4]).astype("float32")
        self.weight_np_0 = np.random.randn(1).astype("float32")
        self.weight_np_1 = np.random.randn(self.x_np.shape[1]).astype("float32")

    def static_check(self, weight_np):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", self.x_np.shape, "float32")
            weight = paddle.static.data("Alpha", weight_np.shape, "float32")
            out = F.prelu(x, weight)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np, "Alpha": weight_np}, fetch_list=[out])
        out_ref = ref_prelu(self.x_np, weight_np)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def dygraph_check(self, weight_np):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        weight = paddle.to_tensor(weight_np)
        out = F.prelu(x, weight)
        out_ref = ref_prelu(self.x_np, weight_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_static_api(self):
        self.static_check(self.weight_np_0)
        # self.static_check(self.weight_np_1)

    def test_dygraph_api(self):
        self.dygraph_check(self.weight_np_0)
        # self.dygraph_check(self.weight_np_1)

    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program()):
            weight_fp32 = paddle.static.data(
                name="weight_fp32", shape=[1], dtype="float32"
            )
            # The input type must be Variable.
            self.assertRaises(TypeError, F.prelu, x=1, weight=weight_fp32)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.static.data(name="x_int32", shape=[2, 3], dtype="int32")
            self.assertRaises(TypeError, F.prelu, x=x_int32, weight=weight_fp32)
            # support the input dtype is float16
            x_fp16 = paddle.static.data(name="x_fp16", shape=[2, 3], dtype="float16")
            F.prelu(x=x_fp16, weight=weight_fp32)


class TestNNPReluAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.x_np = np.ones([1, 2, 3, 4]).astype("float32")

    def test_static_api(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            x = paddle.static.data(name="X", shape=self.x_np.shape, dtype="float32")
            m = paddle.nn.PReLU()
            out = m(x)
            exe = paddle.static.Executor(self.place)
            exe.run(startup_program)
            res = exe.run(train_program, feed={"X": self.x_np}, fetch_list=[out])
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU()
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        # x = paddle.to_tensor(self.x_np)
        # m = paddle.nn.PReLU(num_parameters=self.x_np.shape[1])
        # out = m(x)
        # out_ref = ref_prelu_nn(self.x_np, self.x_np.shape[1], 0.25)
        # np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(init=0.5)
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(weight_attr=base.ParamAttr(name="weight"))
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(
            weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(0.5))
        )
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        paddle.enable_static()


def prelu_api_wrapper(x, weight, data_format="NCHW"):
    weight = weight.reshape([-1])
    return paddle.nn.functional.prelu(x, weight, data_format, name=None)


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class PReluTest(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        self.init_input_shape()
        self.init_attr()
        self.op_type = "prelu"
        self.python_api = prelu_api_wrapper

        x_np = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        # Since zero point in prelu is not differentiable, avoid randomize
        # zero.
        x_np[np.abs(x_np) < 0.005] = 0.02

        if self.attrs == {
            "mode": "all",
            "data_format": "NCHW",
        } or self.attrs == {"mode": "all", "data_format": "NHWC"}:
            alpha_np = np.random.uniform(-1, -0.5, (1))
        elif self.attrs == {"mode": "channel", "data_format": "NCHW"}:
            alpha_np = np.random.uniform(-1, -0.5, [1, self.x_shape[1], 1, 1])
        elif self.attrs == {"mode": "channel", "data_format": "NHWC"}:
            alpha_np = np.random.uniform(-1, -0.5, [1, 1, 1, self.x_shape[-1]])
        else:
            alpha_np = np.random.uniform(-1, -0.5, [1] + self.x_shape[1:])
            # eager check don't support mode = 'all'
            self.eager_mode = False
        alpha_np = alpha_np.astype(self.dtype)

        self.inputs = {"X": x_np, "Alpha": alpha_np}

        # NOTE(zhiqu): reshape inputs['Alpha'] from [1, 100, 1, 1] to [1, 100] + [1]*len(x.shape[2:])
        # since np operands could not be broadcast together with shapes (1,100,2,2,2,3) (1,100,1,1)
        reshaped_alpha = self.inputs["Alpha"]
        if self.attrs == {"mode": "channel", "data_format": "NCHW"}:
            reshaped_alpha = np.reshape(
                self.inputs["Alpha"],
                [1, self.x_shape[1]] + [1] * len(self.x_shape[2:]),
            )
        elif self.attrs == {"mode": "channel", "data_format": "NHWC"}:
            reshaped_alpha = np.reshape(
                self.inputs["Alpha"],
                [1] + [1] * len(self.x_shape[1:-1]) + [self.x_shape[-1]],
            )
        out_np = np.maximum(self.inputs["X"], 0.0)
        out_np = out_np + np.minimum(self.inputs["X"], 0.0) * reshaped_alpha
        assert out_np is not self.inputs["X"]
        self.outputs = {"Out": out_np}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_input_shape(self):
        self.x_shape = [2, 3, 4, 5]

    def init_attr(self):
        self.attrs = {"mode": "all", "data_format": "NCHW"}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Alpha"], "Out")


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAll(PReluTest):
    def init_input_shape(self):
        self.x_shape = [2, 3, 4, 5]

    def init_attr(self):
        self.attrs = {"mode": "all", "data_format": "NCHW"}


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAllNHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [2, 3, 4, 50]

    def init_attr(self):
        self.attrs = {"mode": "all", "data_format": "NHWC"}


def create_test_fp16_class(parent, check_grad=True, atol=1e-3, max_relative_error=0.05):
    class TestPReluFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16Op")
    TestPReluFp16Case.__name__ = cls_name
    globals()[cls_name] = TestPReluFp16Case


create_test_fp16_class(TestModeAll)
create_test_fp16_class(TestModeAllNHWC)


def prelu_t(x, mode, param_attr=None, name=None, data_format="NCHW"):
    helper = base.layer_helper.LayerHelper("prelu", **locals())
    alpha_shape = [1, x.shape[1], 1, 1]
    dtype = helper.input_dtype(input_param_name="x")
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype="float32",
        is_bias=False,
        default_initializer=paddle.nn.initializer.Constant(0.25),
    )
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x, "Alpha": alpha},
        attrs={"mode": mode, "data_format": data_format},
        outputs={"Out": out},
    )
    return out


# error message test if mode is not one of 'all', 'channel', 'element'
class TestModeError(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.x_np = np.ones([1, 2, 3, 4]).astype("float32")

    def test_mode_error(self):
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name="x", shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, "any")
            except Exception as e:
                assert e.args[0].find("InvalidArgument") != -1

    def test_data_format_error1(self):
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name="x", shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, "channel", data_format="N")
            except Exception as e:
                assert e.args[0].find("InvalidArgument") != -1

    def test_data_format_error2(self):
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name="x", shape=[2, 3, 4, 5])
            try:
                y = paddle.static.nn.prelu(x, "channel", data_format="N")
            except ValueError as e:
                pass


if __name__ == "__main__":
    unittest.main()
