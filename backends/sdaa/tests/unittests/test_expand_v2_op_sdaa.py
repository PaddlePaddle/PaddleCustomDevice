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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.base as base
from paddle.base import Program, program_guard
import paddle

paddle.enable_static()


# Situation 1: shape is a list(without tensor)
class TestExpandV2OpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.python_api = paddle.expand
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.init_data()

        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.shape}
        output = np.tile(self.inputs["X"], self.expand_times)
        self.outputs = {"Out": output}

    def init_data(self):
        self.ori_shape = [100]
        self.shape = [100]
        self.expand_times = [1]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestExpandV2OpRank2_DimExpanding(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [120]
        self.shape = [2, 120]
        self.expand_times = [2, 1]


class TestExpandV2OpRank2(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 140]
        self.shape = [12, 140]
        self.expand_times = [12, 1]


class TestExpandV2OpRank3_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)


class TestExpandV2OpRank4(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.shape = (-1, -1, -1, -1)
        self.expand_times = (1, 1, 1, 1)


class TestExpandV2OpRank5(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 1, 15)
        self.shape = (2, -1, 4, -1)
        self.expand_times = (1, 1, 4, 1)


class TestExpandV2OpRank6(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (4, 1, 30)
        self.shape = (2, -1, 4, 30)
        self.expand_times = (2, 1, 4, 1)


class TestExpandV2OpRank7(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (1, 1, 5, 7, 3, 4)
        self.shape = (2, 2, 5, 7, 3, 4)
        self.expand_times = (2, 2, 1, 1, 1, 1)


# Situation 2: shape is a list(with tensor)
class TestExpandV2OpRank1_tensor_attr(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.python_api = paddle.expand
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.init_data()
        expand_shapes_tensor = []
        for index, ele in enumerate(self.expand_shape):
            expand_shapes_tensor.append(
                ("x" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "expand_shapes_tensor": expand_shapes_tensor,
        }
        self.attrs = {"shape": self.infer_expand_shape}
        output = np.tile(self.inputs["X"], self.expand_times)
        self.outputs = {"Out": output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [1]
        self.expand_shape = [100]
        self.infer_expand_shape = [-1]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestExpandV2OpRank2_Corner_tensor_attr(TestExpandV2OpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [1, 1]
        self.expand_shape = [12, 14]
        self.infer_expand_shape = [12, -1]


# Situation 3: shape is a tensor
class TestExpandV2OpRank1_tensor(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.python_api = paddle.expand
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.init_data()

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "Shape": np.array(self.expand_shape).astype("int32"),
        }
        self.attrs = {}
        output = np.tile(self.inputs["X"], self.expand_times)
        self.outputs = {"Out": output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2, 1]
        self.expand_shape = [2, 100]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


# Situation 4: input x is float16
class TestExpandV2OpFloat16(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "expand_v2"
        self.dtype = np.float16
        self.ori_shape = (2, 4, 20)
        self.inputs = {"X": np.random.random(self.ori_shape).astype(self.dtype)}
        self.attrs = {"shape": [2, 4, 20]}
        output = np.tile(self.inputs["X"], (1, 1, 1))
        self.outputs = {"Out": output}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


# Situation 5: input x is float64
class TestExpandV2OpFloat64(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "expand_v2"
        self.dtype = np.double
        self.ori_shape = (2, 4, 20)
        self.inputs = {"X": np.random.random(self.ori_shape).astype(self.dtype)}
        self.attrs = {"shape": [2, 4, 20]}
        output = np.tile(self.inputs["X"], (1, 1, 1))
        self.outputs = {"Out": output}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


# Situation 6: input x is Integer
class TestExpandV2OpInteger(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.python_api = paddle.expand
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.inputs = {"X": np.random.randint(10, size=(2, 4, 5)).astype("int32")}
        self.attrs = {"shape": [2, 4, 5]}
        output = np.tile(self.inputs["X"], (1, 1, 1))
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output_with_place(self.place)


# Situation 7: input x is Bool
class TestExpandV2OpBoolean(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.python_api = paddle.expand
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.inputs = {"X": np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {"shape": [2, 4, 5]}
        output = np.tile(self.inputs["X"], (1, 1, 1))
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output_with_place(self.place)


# Situation 8: input x is Integer
class TestExpandV2OpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.python_api = paddle.expand
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.inputs = {"X": np.random.randint(10, size=(2, 4, 5)).astype("int64")}
        self.attrs = {"shape": [2, 4, 5]}
        output = np.tile(self.inputs["X"], (1, 1, 1))
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestExpandV2Error(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], paddle.CustomPlace("sdaa", 0)
            )
            shape = [2, 2]
            self.assertRaises(TypeError, paddle.tensor.expand, x1, shape)
            x2 = paddle.static.data(name="x2", shape=[-1, 4], dtype="uint8")
            self.assertRaises(TypeError, paddle.tensor.expand, x2, shape)
            x3 = paddle.static.data(name="x3", shape=[-1, 4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tensor.expand, x3, shape)


# Test python API
class TestExpandV2API(unittest.TestCase):
    def test_api(self):
        input = np.random.random([12, 14]).astype("float32")
        x = paddle.static.data(name="x", shape=[12, 14], dtype="float32")

        positive_2 = paddle.tensor.fill_constant([1], "int32", 12)
        expand_shape = paddle.static.data(name="expand_shape", shape=[2], dtype="int32")

        out_1 = paddle.expand(x, shape=[12, 14])
        out_2 = paddle.expand(x, shape=[positive_2, 14])
        out_3 = paddle.expand(x, shape=expand_shape)

        g0 = base.backward.calc_gradient(out_2, x)

        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
        res_1, res_2, res_3 = exe.run(
            base.default_main_program(),
            feed={"x": input, "expand_shape": np.array([12, 14]).astype("int32")},
            fetch_list=[out_1, out_2, out_3],
        )
        assert np.array_equal(res_1, np.tile(input, (1, 1)))
        assert np.array_equal(res_2, np.tile(input, (1, 1)))
        assert np.array_equal(res_3, np.tile(input, (1, 1)))


class TestExpandInferShape(unittest.TestCase):
    def test_shape_with_var(self):
        with program_guard(Program(), Program()):
            x = paddle.static.data(shape=[-1, 1, 3], name="x")
            fake_var = paddle.randn([2, 3])
            target_shape = [-1, paddle.shape(fake_var)[0], paddle.shape(fake_var)[1]]
            out = paddle.expand(x, shape=target_shape)
            self.assertListEqual(list(out.shape), [-1, -1, -1])


# Test python Dygraph API
class TestExpandV2DygraphAPI(unittest.TestCase):
    def test_expand_times_is_tensor(self):
        with paddle.base.dygraph.guard():
            paddle.seed(1)
            a = paddle.rand([2, 5])
            expand_1 = paddle.expand(a, shape=[2, 5])
            np_array = np.array([2, 5])
            expand_2 = paddle.expand(a, shape=np_array)
            np.testing.assert_allclose(expand_1.numpy(), expand_2.numpy())


if __name__ == "__main__":
    unittest.main()
