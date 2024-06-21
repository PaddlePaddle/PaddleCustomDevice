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
# STRICT LIABILITY,OR TORT (INCLUDINGEargs NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
from scipy.special import logit
from scipy.special import expit
import unittest
import numpy as np
import paddle
from op_test_dy import TestDygraphInplace

SEED = 1234


class TestSub(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].subtract_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {"X": self.x, "Y": self.y}
        self.attrs = {}
        self.outputs = {"Out": self.out}
        self.set_np_compare_func()

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = 0

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad_normal(self):
        if self.dtype == np.int64:
            return
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
            max_relative_error=0.006,
            check_dygraph=False,
            check_inplace=True,
        )


class TestSub1(TestSub):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [4, 3, 1, 1]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [1, 1, 4116, 1]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def test_check_output(self):
        return

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
            max_relative_error=0.006,
            check_dygraph=False,
            check_inplace=False,
            compare_static=True,
        )


class TestAtan(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "atan"
        self.python_api = paddle.atan
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(1234)
        x = np.random.uniform(0, 2, [13, 17]).astype(self.dtype)
        out = np.arctan(x)

        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-7, check_inplace=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.05,
            check_inplace=False,
        )


def bce_wrapper(x, label):
    return paddle._C_ops.bce_loss(x, label)


def bce_loss(input, label):
    return -1 * (label * np.log(input) + (1.0 - label) * np.log(1.0 - input))


class BceTest(TestDygraphInplace):
    def setUp(self):
        self.dtype = np.float32
        self.set_sdaa()
        self.init_test_case()
        self.op_type = "bce_loss"
        self.python_api = bce_wrapper
        input_np = np.random.uniform(0.1, 0.8, self.shape).astype("float32")
        label_np = np.random.randint(0, 2, self.shape).astype("float32")
        output_np = bce_loss(input_np, label_np)

        self.inputs = {"X": input_np, "Label": label_np}
        self.outputs = {"Out": output_np}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)

    def init_test_case(self):
        self.shape = [10, 10]


class TestBceLossOpCase1(BceTest):
    def init_test_cast(self):
        self.shape = [2, 3, 4, 5]


class TestBceLossOpCase2(BceTest):
    def init_test_cast(self):
        self.shape = [2, 3, 20]


class TestClip(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].clip_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "clip"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.max_relative_error = 0.006
        self.python_api = paddle.clip
        self.python_inplace_api = self.inplace_func
        self.inputs = {}
        self.input_temp = {}
        self.initTestCase()

        self.attrs = {}
        self.attrs["min"] = self.min
        self.attrs["max"] = self.max
        if "Min" in self.input_temp:
            min_v = self.input_temp["Min"]
        else:
            min_v = self.attrs["min"]

        if "Max" in self.input_temp:
            max_v = self.input_temp["Max"]
        else:
            max_v = self.attrs["max"]

        input = np.random.random(self.shape).astype(self.dtype)
        input[np.abs(input - min_v) < self.max_relative_error] = 0.5
        input[np.abs(input - max_v) < self.max_relative_error] = 0.5
        self.inputs["X"] = input
        self.outputs = {"Out": np.clip(self.inputs["X"], min_v, max_v)}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (1, 10, 10)
        self.max = 0.8
        self.min = 0.3
        self.input_temp["Max"] = np.array([0.8]).astype(self.dtype)
        self.input_temp["Min"] = np.array([0.3]).astype(self.dtype)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=True)


class TestCos(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "cos"
        self.python_api = paddle.cos
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(1234)

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.cos(x)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.007, check_inplace=False
        )


class TestExp(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].exp_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "exp"
        self.python_api = paddle.exp
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(1234)

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        grad_out = np.ones(out.shape).astype(self.dtype)
        self.grad_out = grad_out
        grad_x = self.compute_gradient(grad_out, out)
        self.grad_x = grad_x

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def compute_gradient(self, grad_out, out):
        return grad_out * out

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5, check_inplace=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out],
        )


def ref_hardswish(x, threshold=6.0, scale=6.0, offset=3.0):
    x_dtype = x.dtype
    if x_dtype == "float16":
        x_dtype = "float16"
        x = x.astype("float32")
    return (x * np.minimum(np.maximum(x + offset, 0.0), threshold) / scale).astype(
        x_dtype
    )


def ref_hard_swish_grad(x, threshold=6.0, scale=6.0, offset=3.0):
    dout = np.full_like(x, fill_value=1.0 / x.size)
    tmp = ((x + offset) < threshold).astype(x.dtype)
    dx = dout * (
        ((x + offset) > 0).astype(x.dtype) * (2 * x + offset) * tmp / scale + 1.0 - tmp
    )
    return dx


class TestHardSwish(TestDygraphInplace):
    def setUp(self):
        self.op_type = "hard_swish"
        self.init_dtype()
        self.init_shape()
        self.python_api = paddle.nn.functional.hardswish

        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True

        np.random.seed(1024)
        x = np.random.uniform(-6, 6, self.shape).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02
        out = ref_hardswish(x, threshold, scale, offset)

        # self.x_grad = ref_hard_swish_grad(x, threshold, scale, offset)
        self.inputs = {"X": x}
        # self.attrs = {'threshold': threshold, 'scale': scale, 'offset': offset}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            # user_defined_grads=[self.x_grad],
            check_inplace=False,
        )


class TestHardSwishFP16(TestHardSwish):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)


def ref_leaky_relu(x, alpha=0.01):
    out = np.copy(x)
    out[out < 0] *= alpha
    return out


class TestLeadyRelu(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "leaky_relu"
        self.python_api = paddle.nn.functional.leaky_relu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(1234)

        self.set_inputs()
        self.set_attrs()
        self.set_outputs()

    def set_inputs(self):
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        self.inputs = {"X": x}

    def set_attrs(self):
        self.attrs = {"alpha": 0.01}

    def set_outputs(self):
        alpha = 0.01 if "alpha" not in self.attrs else self.attrs["alpha"]
        out = ref_leaky_relu(self.inputs["X"], alpha)
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ["X"], "Out", max_relative_error=0.006, check_inplace=False
            )
        else:
            self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)


class TestLeadyRelu2(TestLeadyRelu):
    def init_dtype(self):
        self.dtype = np.float16

    def set_attrs(self):
        self.attrs = {"alpha": 0.5}


class TestLeadyRelu3(TestLeadyRelu):
    def set_attrs(self):
        self.attrs = {"alpha": -0.5}


class TestMul(TestDygraphInplace):
    # case 1: (32, 5) * (5, 100) -> (32, 100)
    def config(self):
        self.x_shape = (32, 5)
        self.y_shape = (5, 100)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "mul"
        self.python_api = paddle.matmul
        self.init_dtype()
        self.config()
        np.random.seed(1234)
        self.inputs = {
            "X": np.random.random(self.x_shape).astype(self.dtype),
            "Y": np.random.random(self.y_shape).astype(self.dtype),
        }
        self.outputs = {"Out": np.dot(self.inputs["X"], self.inputs["Y"])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ["X", "Y"], "Out", max_relative_error=0.5, check_inplace=False
        )

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ["Y"],
            "Out",
            no_grad_set=set("X"),
            max_relative_error=0.5,
            check_inplace=False,
        )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            no_grad_set=set("Y"),
            max_relative_error=0.5,
            check_inplace=False,
        )


class TestPow(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "pow"
        self.python_api = paddle.pow
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(1234)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {"X": x}
        self.attrs = {"factor": 3.0}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)


class TestReciprocal(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].reciprocal_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "reciprocal"
        self.python_api = paddle.reciprocal
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_dtype()
        self.init_shape()

        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, self.shape).astype(self.dtype)
        self.out = np.reciprocal(self.x)

        self.inputs = {"X": self.x}
        self.outputs = {"Out": self.out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        dx = -1 * self.out * self.out
        user_defined_grad = [dx]
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.01,
            user_defined_grads=user_defined_grad,
            check_inplace=True,
        )


class TestReciprocalShape1(TestReciprocal):
    def init_shape(self):
        self.shape = [11, 12, 8, 6]


class TestReciprocalShape2(TestReciprocal):
    def init_shape(self):
        self.shape = [120]


class TestReciprocalDouble(TestReciprocal):
    def init_dtype(self):
        self.dtype = np.double

    def test_check_grad(self):
        pass


class TestRelu(TestDygraphInplace):
    def inplace_func(self, args):
        return paddle.nn.functional.relu_(*args)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "relu"
        self.python_api = paddle.nn.functional.relu
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=True)


class TestReluNeg(TestDygraphInplace):
    def inplace_func(self, args):
        return paddle.nn.functional.relu_(*args)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "relu"
        self.python_api = paddle.nn.functional.relu
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.array([0.1, -0.1, -1.0]).astype(self.dtype)
        out = np.array([0.1, 0.0, 0.0]).astype(self.dtype)

        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)


class TestScale(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].scale_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.python_inplace_api = self.inplace_func
        self.init_dtype()
        self.init_input()
        self.attrs = {
            "scale": self.scale,
            "bias": self.bias,
            "bias_after_scale": self.bias_after_scale,
        }
        self.inputs = {"X": self.x}
        self.outputs = {
            "Out": self.x * self.scale
            + (self.bias if self.bias_after_scale else self.scale * self.bias)
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_input(self):
        self.scale = 2.0
        self.bias = 0.0
        self.bias_after_scale = False
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)


class TestScaleOpFp16(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].scale_(*args[1:])

    def setUp(self):
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)
        self.dtype = np.float32
        self.scale = -2.3
        self.x_fp16 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.attrs = {"scale": self.scale, "bias": 0.0, "bias_after_scale": False}
        self.inputs = {"X": self.x_fp16}
        self.outputs = {"Out": self.x_fp16 * self.scale}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2, check_inplace=True)


class TestScaleOpInt64(TestScale):
    def init_dtype(self):
        self.dtype = np.int64

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2, check_inplace=True)


class TestScaleOpInt32(TestScale):
    def init_dtype(self):
        self.dtype = np.int32


class TestScaleOpInf(TestScale):
    def init_input(self):
        self.scale = np.inf
        self.bias = 1.0
        self.bias_after_scale = False
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)


class TestScaleOpNegInf(TestScale):
    def init_input(self):
        self.scale = -np.inf
        self.bias = 1.0
        self.bias_after_scale = False
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)


class TestBiasAfterScale(TestScale):
    def init_input(self):
        self.scale = 2.0
        self.bias = 1.0
        self.bias_after_scale = True
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)


class TestScatterOp(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].scatter_(*args[1:])

    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        self.python_inplace_api = self.inplace_func
        ref_np = np.ones((3, 50)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 50)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        pass


class TestScatterOp0(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].scatter_(*args[1:])

    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        self.python_inplace_api = self.inplace_func
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.attrs = {"overwrite": True}
        self.outputs = {"Out": output_np}
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        pass


class TestScatterOp1(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].scatter_(*args[1:])

    def setUp(self):
        self.dtype = np.float32
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        self.python_inplace_api = self.inplace_func
        ref_np = np.ones((3, 3)).astype("float32")
        zeros_np = np.zeros([2, 3]).astype("float32")
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        self.attrs = {"overwrite": False}
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        pass


class TestScatterOp2(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].scatter_(*args[1:])

    def setUp(self):
        self.dtype = np.float32
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        self.python_inplace_api = self.inplace_func
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        pass


def loss_wrapper(logit, label, normalize=False, ignore_index=-100):
    out = paddle.nn.functional.binary_cross_entropy_with_logits(
        logit=logit, label=label
    )
    return out


class TestSigmoidCrossEntropyWithLogitsOp1(TestDygraphInplace):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = 64
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(self.dtype)
            ),
            "Label": np.random.randint(0, 2, (batch_size, num_classes)).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                paddle.CustomPlace("sdaa", 0), ["X"], "Out", check_inplace=False
            )
        else:
            self.check_grad_with_place(
                paddle.CustomPlace("sdaa", 0),
                ["X"],
                "Out",
                numeric_place=paddle.CPUPlace(),
                check_inplace=False,
            )

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32


class TestSigmoidCrossEntropyWithLogitsOp3(TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = 64
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(self.dtype)
            ),
            "Label": np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}


class TestSigmoidCrossEntropyWithLogitsOp5(TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes])).astype(
                    self.dtype
                )
            ),
            "Label": np.random.uniform(0, 1, tuple(batch_size + [num_classes])).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}


class TestSigmoidCrossEntropyWithLogitsOp6(TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes])).astype(
                    self.dtype
                )
            ),
            "Label": np.random.randint(0, 2, tuple(batch_size + [num_classes])).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}


class TestSigmoid(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "sigmoid"
        self.python_api = paddle.nn.functional.sigmoid
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.01, check_inplace=False
        )


class TestSilu(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "silu"
        self.python_api = paddle.nn.functional.silu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = x / (np.exp(-x) + 1)

        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)


class TestSiluFp16(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "silu"
        self.python_api = paddle.nn.functional.silu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [13, 14]).astype(self.dtype)
        out = x / (1 + np.exp(-x))

        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)


class TestSinKernel(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "sin"
        self.python_api = paddle.sin
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.sin(x)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.007, check_inplace=False
        )


class TestSoftmax(TestDygraphInplace):
    def inplace_func(self, args):
        return paddle.nn.functional.softmax_(*args)

    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "softmax"
        self.python_api = paddle.nn.functional.softmax
        self.python_inplace_api = self.inplace_func
        self.init_dtype()

        x = np.random.random([3, 3]).astype(self.dtype)
        np_out = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": np_out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
            atol=1e-3 if self.dtype == np.float16 else 1e-5,
            check_inplace=True,
        )

    # TODO(zhanggq): inplace version of tecodnnSoftmaxBackward has precision loss, to be fixed by tecodnn
    # def test_check_grad_no_input(self):
    #     if self.dtype == np.float16:
    #         self.check_grad_with_place(paddle.CustomPlace('sdaa', 0), ['X'],
    #                                    'Out',
    #                                    max_relative_error=0.01,
    #                                    check_inplace=True)
    #     else:
    #         self.check_grad_with_place(paddle.CustomPlace('sdaa', 0), ['X'],
    #                                    'Out',
    #                                    numeric_place=paddle.CPUPlace(),
    #                                    check_inplace=True)


def stable_softmax(x):
    # Compute the softmax of vector x in a numerically stable way.
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1 :]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def python_core_api(
    logits,
    label,
    soft_label=False,
    ignore_index=-100,
    numeric_stable_mode=True,
    use_softmax=False,
    axis=-1,
):
    # the API paddle.nn.functional.softmax_with_cross_entropy cannot
    # set use_softmax=False, so add a core api manually
    # assert use_softmax is False
    softmax, loss = paddle._C_ops.cross_entropy_with_softmax(
        logits, label, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis
    )
    if use_softmax is False:
        return loss
    else:
        return loss, softmax


class TestSoftmaxWithCrossEntropyOp(TestDygraphInplace):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = False
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True
        np.random.seed(SEED)

    def init_logits(self):
        self.logits = getattr(
            self, "logits", np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        )
        self.softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

    def init_label(self):
        if self.soft_label:
            self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            self.labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

    def setUp(self):
        self.set_sdaa()
        self.op_type = "softmax_with_cross_entropy"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_params()
        self.init_logits()
        self.init_label()

        loss = cross_entropy(
            self.softmax, self.labels, self.soft_label, self.axis, self.ignore_index
        )
        self.outputs = {
            "Loss": loss.astype(self.dtype),
        }
        if self.use_softmax == False:  # noqa
            self.inputs = {"Logits": self.softmax, "Label": self.labels}
        else:
            self.inputs = {"Logits": self.logits, "Label": self.labels}
            self.outputs["Softmax"] = self.softmax.astype(self.dtype)

        self.attrs = {
            "soft_label": self.soft_label,
            "ignore_index": self.ignore_index,
            "numeric_stable_mode": self.numeric_stable_mode,
            "use_softmax": self.use_softmax,
        }

        if self.axis != -1:
            self.attrs["axis"] = self.axis

    def test_check_grad(self):
        # fp32 has low precision, cpu and sdaa both need to relax the max_relative_error if using fp32
        self.check_grad_with_place(
            self.place,
            ["Logits"],
            "Loss",
            numeric_grad_delta=0.001,
            max_relative_error=0.5,
            check_inplace=False,
        )


class TestSoftmaxWithCrossEntropyOpNoSoftmax(TestSoftmaxWithCrossEntropyOp):
    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = False
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = False
        np.random.seed(SEED)


class TestSoftmaxWithCrossEntropyOpOneHot(TestSoftmaxWithCrossEntropyOp):
    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = True
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True
        np.random.seed(SEED)

    # initialize one hot label
    def init_label(self):
        batch_size, class_num = self.shape
        self.label_index = np.random.randint(0, class_num, (batch_size))
        self.labels = np.zeros(self.logits.shape).astype(self.dtype)
        self.labels[np.arange(batch_size), self.label_index] = 1


class TestSoftmaxWithCrossEntropyOpSoftLabel(TestSoftmaxWithCrossEntropyOp):
    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = True
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True
        np.random.seed(SEED)


def ref_softplus(x, beta=1, threshold=20):
    x_beta = beta * x
    out = np.select(
        [x_beta <= threshold, x_beta > threshold],
        [np.log(1 + np.exp(x_beta)) / beta, x],
    )
    return out


class TestSoftplus(TestDygraphInplace):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "softplus"
        self.python_api = paddle.nn.functional.softplus
        self.init_dtype()
        self.init_shape()

        beta = 2
        threshold = 15

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_softplus(x, beta, threshold)
        self.inputs = {"X": x}
        self.attrs = {"beta": beta, "threshold": threshold}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)

    def init_dtype(self):
        self.dtype = np.float32


class TestSoftplusFp16(TestSoftplus):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.006, check_inplace=False
        )


def ref_softsign(x):
    out = np.divide(x, 1 + np.abs(x))
    return out


class TestSoftsign(TestDygraphInplace):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "softsign"
        self.python_api = paddle.nn.functional.softsign
        self.init_dtype()
        self.init_shape()

        np.random.seed(SEED)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = ref_softsign(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=False)

    def init_dtype(self):
        self.dtype = np.float32


class TestSoftsignFp16(TestSoftsign):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.006, check_inplace=False
        )


class TestSqrt(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].sqrt_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "sqrt"
        self.python_api = paddle.sqrt
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ["X"], "Out", max_relative_error=0.009, check_inplace=True
            )
        else:
            self.check_grad_with_place(
                self.place, ["X"], "Out", max_relative_error=0.009, check_inplace=True
            )


@unittest.skip("Sqrt_ not support fp16")
class TestSqrtFp16(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].sqrt_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "sqrt"
        self.python_api = paddle.sqrt
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3, check_inplace=True)


class TestSquareKernel(TestDygraphInplace):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "square"
        self.python_api = paddle.square
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)

        x = np.random.uniform(-1, 1, [11, 17, 16, 8]).astype(self.dtype)
        out = np.square(x, dtype=self.dtype)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.007, check_inplace=False
        )


class TestCase2(TestSquareKernel):
    def init_dtype(self):
        self.dtype = np.float32


# Squeeze_ cannot support
class TestSqueeze2Op(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].squeeze_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "squeeze2"
        self.python_api = paddle.squeeze
        self.python_inplace_api = self.inplace_func
        self.dtype = np.float32
        self.python_out_sig = ["Out"]  # python out sig is customized output signature.
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_check_output(self):
        self.check_output_with_place(paddle.CustomPlace("sdaa", 0), check_inplace=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CustomPlace("sdaa", 0), ["X"], "Out", check_inplace=False
        )

    @unittest.skip("squeeze_ backward error")
    def test_check_grad_inplace(self):
        self.check_grad_with_place(
            paddle.CustomPlace("sdaa", 0), ["X"], "Out", check_inplace=True
        )

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestSqueeze2Op1(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


# Correct: No axes input.
class TestSqueeze2Op2(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed.
class TestSqueeze2Op3(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


class TestTanh(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].tanh_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "tanh"
        self.python_api = paddle.tanh
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(-1, 1, [10, 20]).astype(self.dtype)
        out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=True)


class TestTanhFp16(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].tanh_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "tanh"
        self.python_api = paddle.tanh
        self.python_inplace_api = self.inplace_func
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        out = np.tanh(x)

        self.inputs = {"X": x}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3, check_inplace=True)


class TestUnsqueeze2Op(TestDygraphInplace):
    def inplace_func(self, args):
        return args[0].unsqueeze_(*args[1:])

    def setUp(self):
        self.set_sdaa()
        self.op_type = "unsqueeze2"
        self.python_api = paddle.unsqueeze
        self.python_inplace_api = self.inplace_func
        self.python_out_sig = ["Out"]
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_test_case()
        self.dtype = np.float32
        self.x = np.random.random(self.ori_shape).astype("float32")
        self.inputs = {"X": self.x}
        self.init_attrs()
        self.outputs = {
            "Out": self.x.reshape(self.new_shape),
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_inplace=True)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_inplace=True)

    def init_test_case(self):
        self.ori_shape = (3, 40)
        self.axes = (0, 2)
        self.new_shape = (1, 3, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestUnsqueeze2Op1(TestUnsqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (0, -2)
        self.new_shape = (1, 20, 1, 5)


# Correct: No axes input.
class TestUnsqueeze2Op2(TestUnsqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = ()
        self.new_shape = (10, 2, 5)


# Correct: Just part of axes be squeezed.
class TestUnsqueeze2Op3(TestUnsqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (6, 5, 1, 4)
        self.axes = (1, -1)
        self.new_shape = (6, 1, 5, 1, 4, 1)


if __name__ == "__main__":
    unittest.main()
