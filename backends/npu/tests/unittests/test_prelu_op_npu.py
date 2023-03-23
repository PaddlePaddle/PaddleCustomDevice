#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid.framework import _test_eager_guard
from tests.op_test import OpTest, skip_check_grad_ci

paddle.enable_static()


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
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)
        self.x_np = np.random.uniform(-1.0, 1.0, [1, 2, 3, 4]).astype("float32")
        self.weight_np_0 = np.random.randn(1).astype("float32")
        self.weight_np_1 = np.random.randn(self.x_np.shape[1]).astype("float32")

    def static_check(self, weight_np):
        with paddle.static.program_guard(paddle.static.Program()):
            paddle.enable_static()
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
        self.static_check(self.weight_np_1)

    def test_dygraph_api(self):
        self.dygraph_check(self.weight_np_0)
        self.dygraph_check(self.weight_np_1)

    def test_dygraph_api_eager(self):
        with _test_eager_guard():
            self.test_dygraph_api()


class TestNNPReluAPI(unittest.TestCase):
    def setUp(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)
        self.x_np = np.ones([1, 2, 3, 4]).astype("float32")

    def test_static_api(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            paddle.enable_static()
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

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(num_parameters=self.x_np.shape[1])
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, self.x_np.shape[1], 0.25)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(init=0.5)
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(weight_attr=fluid.ParamAttr(name="weight"))
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(
            weight_attr=fluid.ParamAttr(initializer=paddle.nn.initializer.Constant(0.5))
        )
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        paddle.enable_static()


def prelu_api_wrapper(x, weight, data_format="NCHW"):
    weight = weight.reshape([-1])
    return paddle.nn.functional.prelu(x, weight, data_format, name=None)


class PReluTest(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.init_dtype()
        self.init_input_shape()
        self.init_place()
        self.eager_mode = True
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

    def init_dtype(self):
        self.dtype = np.float32

    def init_input_shape(self):
        self.x_shape = [2, 100, 3, 4]

    def init_place(self):
        self.place = paddle.CustomPlace("npu", 0)

    def init_attr(self):
        self.attrs = {"mode": "channel", "data_format": "NCHW"}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X", "Alpha"], "Out", numeric_place=paddle.CPUPlace()
        )


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAll(PReluTest):
    def init_input_shape(self):
        self.x_shape = [2, 3, 4, 50]

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


class TestModeElt(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 2, 5, 10]

    def init_attr(self):
        self.attrs = {"mode": "element", "data_format": "NCHW"}


class TestModeEltNHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 2, 5, 10]

    def init_attr(self):
        self.attrs = {"mode": "element", "data_format": "NHWC"}


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAllRank3(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        self.attrs = {"mode": "all", "data_format": "NCHW"}

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Alpha"],
            "Out",
            max_relative_error=0.006,
            numeric_place=paddle.CPUPlace(),
        )


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAllRank3NHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        self.attrs = {"mode": "all", "data_format": "NHWC"}

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Alpha"],
            "Out",
            max_relative_error=0.006,
            numeric_place=paddle.CPUPlace(),
        )


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAllRank6(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 2, 3, 4, 5, 6]

    def init_attr(self):
        self.attrs = {"mode": "all", "data_format": "NCHW"}


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAllRank6NHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 2, 3, 4, 5, 6]

    def init_attr(self):
        self.attrs = {"mode": "all", "data_format": "NHWC"}


class TestModeChannelRank3(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        self.attrs = {"mode": "channel", "data_format": "NCHW"}


class TestModeChannelRank3NHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 3, 100]

    def init_attr(self):
        self.attrs = {"mode": "channel", "data_format": "NHWC"}


class TestModeChannelRank6(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 100, 2, 2, 2, 2]

    def init_attr(self):
        self.attrs = {"mode": "channel", "data_format": "NCHW"}


class TestModeChannelRank6NHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 2, 2, 2, 2, 100]

    def init_attr(self):
        self.attrs = {"mode": "channel", "data_format": "NHWC"}


class TestModeChannelNHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 3, 1, 100]

    def init_attr(self):
        self.attrs = {"mode": "channel", "data_format": "NHWC"}


class TestModeElementRank3(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 10, 10]

    def init_attr(self):
        self.attrs = {"mode": "element", "data_format": "NCHW"}


class TestModeElementRank3NHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 10, 10]

    def init_attr(self):
        self.attrs = {"mode": "element", "data_format": "NHWC"}


class TestModeElementRank6(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 2, 2, 4, 5, 2]

    def init_attr(self):
        self.attrs = {"mode": "element", "data_format": "NCHW"}


class TestModeElementRank6NHWC(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 2, 2, 4, 5, 2]

    def init_attr(self):
        self.attrs = {"mode": "element", "data_format": "NHWC"}


def create_test_fp16_class(parent, check_grad=True, atol=1e-3, max_relative_error=0.05):
    class TestPReluFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_grad(self):
            self.check_grad_with_place(
                self.place, ["X", "Alpha"], "Out", max_relative_error=max_relative_error
            )

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16Op")
    TestPReluFp16Case.__name__ = cls_name
    globals()[cls_name] = TestPReluFp16Case


create_test_fp16_class(TestModeElt)
create_test_fp16_class(TestModeAllRank3)
create_test_fp16_class(TestModeAllRank6)
create_test_fp16_class(TestModeChannelRank3)
create_test_fp16_class(TestModeChannelRank6)
create_test_fp16_class(TestModeElementRank3)
create_test_fp16_class(TestModeElementRank6)
create_test_fp16_class(TestModeEltNHWC)
create_test_fp16_class(TestModeAllRank3NHWC)
create_test_fp16_class(TestModeAllRank6NHWC)
create_test_fp16_class(TestModeChannelNHWC)
create_test_fp16_class(TestModeChannelRank3NHWC)
create_test_fp16_class(TestModeChannelRank6NHWC)
create_test_fp16_class(TestModeElementRank3NHWC)
create_test_fp16_class(TestModeElementRank6NHWC)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
