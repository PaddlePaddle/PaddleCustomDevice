# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest

import numpy as np
from scipy.special import erf

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from tests.op_test import OpTest

paddle.enable_static()
SEED = 2021


class TestActivation(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "exp"
        self.init_dtype()
        self.init_shape()
        self.init_kernel_type()
        self.check_eager = True
        self.python_api = paddle.exp

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_output(self):
        check_eager = False
        if hasattr(self, "check_eager"):
            check_eager = self.check_eager
        self.check_output_with_place(self.place, check_eager=check_eager)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        check_eager = False
        if hasattr(self, "check_eager"):
            check_eager = self.check_eager
        self.check_grad_with_place(self.place, ["X"], "Out", check_eager=check_eager)

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def init_kernel_type(self):
        pass


class TestActivation_ZeroDim(TestActivation):
    def init_shape(self):
        self.shape = []


class TestAtan(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "atan"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.arctan(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out")

    def test_out_name(self):
        with fluid.program_guard(fluid.Program()):
            np_x = np.array([0.1])
            data = fluid.layers.data(name="X", shape=[1])
            out = paddle.atan(data, name="Y")
            exe = fluid.Executor(self.place)
            (result,) = exe.run(feed={"X": np_x}, fetch_list=[out])
            expected = np.arctan(np_x)
            self.assertEqual(result, expected)

    def test_dygraph(self):
        with fluid.dygraph.guard(self.place):
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = paddle.atan(x).numpy()
            z_expected = np.arctan(np_x)
            self.assertEqual(z, z_expected)


class TestAtan_ZeroDim(TestAtan):
    def init_shape(self):
        self.shape = []


class TestCos(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "cos"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.cos(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.01)


class TestCos_ZeroDim(TestCos):
    def init_shape(self):
        self.shape = []


def ref_leaky_relu(x, alpha=0.01):
    out = np.copy(x)
    out[out < 0] *= alpha
    return out


class TestLeakyRelu(TestActivation):
    def get_alpha(self):
        return 0.02

    def setUp(self):
        self.set_npu()
        self.op_type = "leaky_relu"
        self.init_dtype()
        self.init_shape()
        alpha = self.get_alpha()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.05
        out = ref_leaky_relu(x, alpha)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        self.attrs = {"alpha": alpha}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestLeakyReluAlpha1(TestLeakyRelu):
    def get_alpha(self):
        return 2


class TestLeakyReluAlpha2(TestLeakyRelu):
    def get_alpha(self):
        return -0.01


class TestLeakyReluAlpha3(TestLeakyRelu):
    def get_alpha(self):
        return -2.0


class TestLeakyRelu_ZeroDim(TestLeakyRelu):
    def init_shape(self):
        self.shape = []


class TestLeakyReluAPI(unittest.TestCase):
    # test paddle.nn.LeakyReLU, paddle.nn.functional.leaky_relu,
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype("float32")
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", [10, 12])
            out1 = F.leaky_relu(x)
            m = paddle.nn.LeakyReLU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_leaky_relu(self.x_np)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.leaky_relu(x)
        m = paddle.nn.LeakyReLU()
        out2 = m(x)
        out_ref = ref_leaky_relu(self.x_np)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

        out1 = F.leaky_relu(x, 0.6)
        m = paddle.nn.LeakyReLU(0.6)
        out2 = m(x)
        out_ref = ref_leaky_relu(self.x_np, 0.6)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


def gelu(x, approximate):
    if approximate:
        y_ref = (
            0.5
            * x
            * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


@unittest.skip(reason="NPU GELU do not support approximate is True")
class TestGeluApproximate(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "gelu"
        self.init_dtype()
        self.init_shape()
        approximate = True
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        self.attrs = {"approximate": approximate}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestGelu(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "gelu"
        self.init_dtype()
        self.init_shape()
        approximate = False
        np.random.seed(2048)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        self.attrs = {"approximate": approximate}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestGelu_ZeroDim(TestGelu):
    def init_shape(self):
        self.shape = []


class TestGELUAPI(unittest.TestCase):
    # test paddle.nn.GELU, paddle.nn.functional.gelu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype("float32")
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", [11, 17])
            out1 = F.gelu(x)
            m = paddle.nn.GELU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = gelu(self.x_np, False)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-03)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.gelu(x)
        m = paddle.nn.GELU()
        out2 = m(x)
        out_ref = gelu(self.x_np, False)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-03)

        out1 = F.gelu(x, True)
        m = paddle.nn.GELU(True)
        out2 = m(x)
        out_ref = gelu(self.x_np, True)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


def celu(x, alpha):
    out_ref = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return out_ref.astype(x.dtype)


class TestCELU(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "celu"
        self.init_dtype()
        self.init_shape()

        self.python_api = paddle.nn.functional.celu
        np.random.seed(1024)
        x = np.random.uniform(-3, 3, self.shape).astype(self.dtype)
        alpha = 1.5
        out = celu(x, alpha)
        self.inputs = {"X": x}
        self.attrs = {"alpha": alpha}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out", check_eager=True)


class TestCELU_ZeroDim(TestCELU):
    def init_shape(self):
        self.shape = []


class TestCELUAPI(unittest.TestCase):
    # test paddle.nn.CELU, paddle.nn.functional.celu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype("float32")
        self.place = paddle.CustomPlace("npu", 0)
        self.executed_api()

    def executed_api(self):
        self.celu = F.celu

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", [10, 12])
            out1 = self.celu(x, 1.5)
            m = paddle.nn.CELU(1.5)
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = celu(self.x_np, 1.5)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = self.celu(x, 1.5)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.CELU(1.5)
        out2 = m(x)
        out_ref = celu(self.x_np, 1.5)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

        out1 = self.celu(x, 0.2)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.CELU(0.2)
        out2 = m(x)
        out_ref = celu(self.x_np, 0.2)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


class TestLog(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "log"
        self.check_eager = True
        self.python_api = paddle.log
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.log(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out", check_eager=True)


class TestLog_ZeroDim(TestLog):
    def init_shape(self):
        self.shape = []


class TestPow(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "pow"
        self.python_api = paddle.pow
        self.check_eager = True
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, self.shape).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {"factor": 3.0}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=self.check_eager)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["X"], "Out", check_eager=self.check_eager
        )


class TestPow_ZeroDim(TestPow):
    def init_shape(self):
        self.shape = []


class TestPow_factor_tensor(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "pow"
        self.check_eager = False
        self.python_api = paddle.pow
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {
            "X": OpTest.np_dtype_to_fluid_dtype(x),
            "FactorTensor": np.array([3.0]).astype("float32"),
        }

        self.attrs = {}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=self.check_eager)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["X"], "Out", check_eager=self.check_eager
        )


class TestRelu(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "relu"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)
        self.inputs = {"X": x}

        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestRelu_ZeroDim(TestRelu):
    def init_shape(self):
        self.shape = []


class TestReluAPI(unittest.TestCase):
    # test paddle.nn.ReLU, paddle.nn.functional.relu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype("float32")
        self.place = paddle.CustomPlace("npu", 0)
        self.executed_api()

    def executed_api(self):
        self.relu = F.relu

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", [10, 12])
            out1 = self.relu(x)
            m = paddle.nn.ReLU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = np.maximum(self.x_np, 0)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.ReLU()
        out1 = m(x)
        out2 = self.relu(x)
        out_ref = np.maximum(self.x_np, 0)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


class TestReluInplaceAPI(TestReluAPI):
    # test paddle.nn.functional.relu_
    def executed_api(self):
        self.relu = F.relu_


def ref_relu6(x, threshold=6.0):
    out = np.copy(x)
    out[np.abs(x - threshold) < 0.005] = threshold + 0.02
    out = np.minimum(np.maximum(x, 0), threshold)
    return out


class TestRelu6(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "relu6"
        self.init_dtype()
        self.init_shape()
        self.python_api = paddle.nn.functional.relu6

        np.random.seed(1024)
        x = np.random.uniform(-1, 10, self.shape).astype(self.dtype)
        x[np.abs(x) < 0.005] = 0.02
        out = ref_relu6(x)

        self.inputs = {"X": x}
        self.attrs = {"threshold": 6.0}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out", check_eager=True)


class TestRelu6_ZeroDim(TestRelu6):
    def init_shape(self):
        self.shape = []


class TestRelu6API(unittest.TestCase):
    # test paddle.nn.ReLU6, paddle.nn.functional.relu6
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 10, [10, 12]).astype(np.float64)
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.relu6(x)
            relu6 = paddle.nn.ReLU6()
            out2 = relu6(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_relu6(self.x_np)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.relu6(x)
        relu6 = paddle.nn.ReLU6()
        out2 = relu6(x)
        out_ref = ref_relu6(self.x_np)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data("X", self.x_np.shape, self.x_np.dtype)
            out = paddle.nn.functional.relu6(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out])
        out_ref = ref_relu6(self.x_np)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)


class TestSquare(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "square"
        self.python_api = paddle.square
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.square(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.007, check_eager=True
        )

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=True)


class TestSquare_ZeroDim(TestSquare):
    def init_shape(self):
        self.shape = []


class TestSqrt(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "sqrt"
        self.python_api = paddle.sqrt
        self.init_dtype()
        self.init_shape()

        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out", check_eager=True)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=True)


class TestSqrt_ZeroDim(TestSqrt):
    def init_shape(self):
        self.shape = []


class TestSigmoid(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "sigmoid"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.01)


class TestSigmoid_ZeroDim(TestSigmoid):
    def init_shape(self):
        self.shape = []


class TestTanh(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "tanh"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.tanh(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestTanh_ZeroDim(TestTanh):
    def init_shape(self):
        self.shape = []


class TestTanhAPI(unittest.TestCase):
    # test paddle.tanh, paddle.nn.tanh, paddle.nn.functional.tanh
    def setUp(self):
        self.dtype = "float32"
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        self.place = paddle.CustomPlace("npu", 0)
        self.executed_api()

    def executed_api(self):
        self.tanh = F.tanh

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", [10, 12], self.dtype)
            out1 = self.tanh(x)
            th = paddle.nn.Tanh()
            out2 = th(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = np.tanh(self.x_np)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.tanh(x)
        out2 = paddle.tanh(x)
        th = paddle.nn.Tanh()
        out3 = th(x)
        out_ref = np.tanh(self.x_np)
        for r in [out1, out2, out3]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


class TestTanhInplaceAPI(TestTanhAPI):
    # test paddle.tanh_
    def executed_api(self):
        self.tanh = paddle.tanh_


class TestFloor(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "floor"
        self.check_eager = True
        self.python_api = paddle.floor
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.floor(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    # the gradient on floor, ceil, round is undefined.
    # we return zero as gradient, but the numpy return nan
    # The same reason with TestFloor
    def test_check_grad(self):
        pass


class TestFloor_ZeroDim(TestFloor):
    def init_shape(self):
        self.shape = []


def ref_softshrink(x, threshold=0.5):
    out = np.copy(x)
    out = (out < -threshold) * (out + threshold) + (out > threshold) * (out - threshold)
    return out


class TestSoftShrink(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "softshrink"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_dtype()
        self.init_shape()

        x = np.random.uniform(-5, 5, self.shape).astype(self.dtype)
        threshold = 0.5

        out = ref_softshrink(x, threshold)
        self.inputs = {"X": x}
        self.attrs = {"threshold": threshold}
        self.outputs = {"Out": out}

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestSoftShrinkFp16(TestSoftShrink):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestSoftshrink_ZeroDim(TestSoftShrink):
    def init_shape(self):
        self.shape = []


class TestSoftshrinkAPI(unittest.TestCase):
    # test paddle.nn.Softshrink, paddle.nn.functional.softshrink
    def setUp(self):
        self.threshold = 0.8
        np.random.seed(1024)
        self.x_np = np.random.uniform(0.25, 10, [10, 12]).astype(np.float64)
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.softshrink(x, self.threshold)
            softshrink = paddle.nn.Softshrink(self.threshold)
            out2 = softshrink(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_softshrink(self.x_np, self.threshold)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.softshrink(x, self.threshold)
        softshrink = paddle.nn.Softshrink(self.threshold)
        out2 = softshrink(x)
        out_ref = ref_softshrink(self.x_np, self.threshold)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


def ref_softplus(x, beta=1, threshold=20):
    x_beta = beta * x
    out = np.select(
        [x_beta <= threshold, x_beta > threshold],
        [np.log(1 + np.exp(x_beta)) / beta, x],
    )
    return out


class TestSoftplus(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "softplus"
        self.init_dtype()
        self.init_shape()

        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        beta = 2
        threshold = 15

        out = ref_softplus(x, beta, threshold)
        self.inputs = {"X": x}
        self.attrs = {"beta": beta, "threshold": threshold}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestSoftplus_ZeroDim(TestSoftplus):
    def init_shape(self):
        self.shape = []


class TestSoftplusFp16(TestSoftplus):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestSoftplusAPI(unittest.TestCase):
    # test paddle.nn.Softplus, paddle.nn.functional.softplus
    def setUp(self):
        self.beta = 2
        self.threshold = 15
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.softplus(x, self.beta, self.threshold)
            softplus = paddle.nn.Softplus(self.beta, self.threshold)
            out2 = softplus(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_softplus(self.x_np, self.beta, self.threshold)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.softplus(x, self.beta, self.threshold)
        softplus = paddle.nn.Softplus(self.beta, self.threshold)
        out2 = softplus(x)
        out_ref = ref_softplus(self.x_np, self.beta, self.threshold)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


class TestSin(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "sin"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    @unittest.skip("sin_grad not implemented on NPU yet")
    def test_check_grad(self):
        pass


class TestSin_ZeroDim(TestSin):
    def init_shape(self):
        self.shape = []


# ------------------ Test Fp16 ----------------------
def create_test_act_fp16_class(parent, atol=1e-3, grad_check=True, grad_atol=0.80):
    class TestActFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            self.check_output_with_place(self.place, atol=atol)

        def test_check_grad(self):
            if grad_check:
                self.check_grad_with_place(
                    self.place, ["X"], "Out", max_relative_error=grad_atol
                )

    cls_name = "{0}_{1}".format(parent.__name__, "fp16")
    TestActFp16.__name__ = cls_name
    globals()[cls_name] = TestActFp16


create_test_act_fp16_class(TestActivation)
create_test_act_fp16_class(TestLeakyRelu)
create_test_act_fp16_class(TestCos, grad_atol=0.85)
create_test_act_fp16_class(TestRelu)
create_test_act_fp16_class(TestRelu6)
create_test_act_fp16_class(TestGelu)
create_test_act_fp16_class(TestCELU)
create_test_act_fp16_class(TestAtan)
create_test_act_fp16_class(TestSin, grad_check=False)
create_test_act_fp16_class(TestSqrt)
create_test_act_fp16_class(TestSquare)
create_test_act_fp16_class(TestSigmoid)
create_test_act_fp16_class(TestTanh)
create_test_act_fp16_class(TestLog, atol=1e-2)
create_test_act_fp16_class(TestPow, atol=5e-2)
create_test_act_fp16_class(TestPow_factor_tensor, atol=5e-2)
create_test_act_fp16_class(TestFloor, grad_check=False)

# TODO(qili93): merge elu, swish ops into activaions

if __name__ == "__main__":
    unittest.main()
