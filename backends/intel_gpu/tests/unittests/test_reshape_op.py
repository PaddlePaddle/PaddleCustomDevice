#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np

from op_test import OpTest
import paddle
import paddle.base as base
from paddle.static import Program, program_guard


def get_places(self):
    return [paddle.CustomPlace("intel_gpu", 0)]


paddle.enable_static()
OpTest._get_places = get_places


# situation 1: have shape( list, no tensor), no actual shape(Tensor)
class TestReshapeOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.infered_shape = (12, 10)

    def test_check_output(self):
        self.check_output(no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (5, 25)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshapeOpDimInfer2(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)


# situation 2: have shape(list, no tensor), have actual shape(Tensor)
class TestReshapeOpWithInputShape(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "Shape": np.array(self.actual_shape, dtype="int32"),
        }
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.actual_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (6, 20)
        self.new_shape = (0, -1, 20)
        self.actual_shape = (2, 3, 20)

    def test_check_output(self):
        self.check_output(no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Situation 3: have shape(list, have tensor), no actual shape(Tensor)
class TestReshapeOp_attr_ShapeTensor(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"

        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(("x" + str(index), np.ones((1)).astype("int32") * ele))

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "ShapeTensor": shape_tensor,
        }
        self.attrs = {"shape": self.shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.infered_shape = (10, 10)
        self.shape = (-1, -1)

    def test_check_output(self):
        self.check_output(no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 20)
        self.infered_shape = (5, -1, 20)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)


# Situation 4: have shape(Tensor), no actual shape(Tensor)
class TestReshapeOp_attr_OnlyShape(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "Shape": np.array(self.new_shape, dtype="int32"),
        }
        self.attrs = {}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def init_data(self):
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.infered_shape = (10, 10)

    def test_check_output(self):
        self.check_output(no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 10)
        self.infered_shape = (5, -1, 10)
        self.shape = (5, -1, -1)


class TestReshapeOpDimInfer2_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)


class TestReshapeOpBool(TestReshapeOp):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.inputs = {"X": np.random.choice([True, False], size=self.ori_shape)}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def test_check_grad(self):
        pass


# Test python API
class TestReshapeAPI(unittest.TestCase):
    def _set_paddle_api(self):
        self.fill_constant = paddle.base.layers.fill_constant
        self.data = paddle.static.data
        self.to_tensor = paddle.to_tensor
        self._executed_api()

    def _executed_api(self):
        self.reshape = paddle.reshape

    def _set_base_api(self):
        self.fill_constant = base.layers.fill_constant
        self.data = paddle.static.data
        self.reshape = base.layers.reshape

    def _test_api(self):
        paddle.enable_static()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        main_prog = Program()
        with program_guard(main_prog, Program()):
            positive_five = self.fill_constant([1], "int32", 5)
            x = self.data(name="x", shape=[2, 25], dtype="float32")

            actual_shape = self.data(name="shape", shape=[3], dtype="int32")

            # situation 1: have shape( list, no tensor), no actual shape(Tensor)
            out_1 = self.reshape(x, shape)

            # situation 2: have shape(list, no tensor), have actual shape(Tensor)
            out_2 = base.layers.reshape(x, shape=shape, actual_shape=actual_shape)

            # Situation 3: have shape(list, have tensor), no actual shape(Tensor)
            out_3 = self.reshape(x, shape=[positive_five, 10])

            # Situation 4: have shape(Tensor), no actual shape(Tensor)
            out_4 = self.reshape(x, shape=actual_shape)

        exe = paddle.static.Executor(place=paddle.CustomPlace("intel_gpu", 0))
        res_1, res_2, res_3, res_4 = exe.run(
            main_prog,
            feed={"x": input, "shape": np.array([2, 5, 5]).astype("int32")},
            fetch_list=[out_1, out_2, out_3, out_4],
        )

        assert np.array_equal(res_1, input.reshape(shape))
        assert np.array_equal(res_2, input.reshape(shape))
        assert np.array_equal(res_3, input.reshape([5, 10]))
        assert np.array_equal(res_4, input.reshape(shape))

    def test_paddle_api(self):
        self._set_paddle_api()
        self._test_api()

    def test_base_api(self):
        self._set_base_api()
        self._test_api()

    def test_imperative(self):
        self._set_paddle_api()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        with base.dygraph.guard(paddle.CustomPlace("intel_gpu", 0)):
            x = self.to_tensor(input)
            positive_five = self.fill_constant([1], "int32", 5)

            out_1 = self.reshape(x, shape)

            out_2 = self.reshape(x, shape=[positive_five, 10])

            shape_tensor = self.to_tensor(np.array([2, 5, 5]).astype("int32"))
            out_3 = self.reshape(x, shape=shape_tensor)

        assert np.array_equal(out_1.numpy(), input.reshape(shape))
        assert np.array_equal(out_2.numpy(), input.reshape([5, 10]))
        assert np.array_equal(out_3.numpy(), input.reshape(shape))


class TestStaticReshape_(TestReshapeAPI):
    def _executed_api(self):
        self.reshape = paddle.reshape_

    def test_imperative(self):
        self._set_paddle_api()
        input = np.random.random([2, 25]).astype("float32")
        shape = [2, 5, 5]
        with base.dygraph.guard(paddle.CustomPlace("intel_gpu", 0)):
            x = self.to_tensor(input)
            positive_five = self.fill_constant([1], "int32", 5)

            out_1 = self.reshape(x, shape)

            out_2 = self.reshape(x, shape=[positive_five, 10])

            shape_tensor = self.to_tensor(np.array([2, 5, 5]).astype("int32"))
            out_3 = self.reshape(x, shape=shape_tensor)

        assert np.array_equal(out_1.numpy(), input.reshape(shape))
        assert np.array_equal(out_2.numpy(), input.reshape(shape))
        assert np.array_equal(out_3.numpy(), input.reshape(shape))


class TestDygraphReshapeAPI(unittest.TestCase):
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.reshape = paddle.reshape

    def test_out(self):
        paddle.disable_static(paddle.CustomPlace("intel_gpu", 0))
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        self.assertTrue(np.allclose(expected_out, out_np))

    def test_out_uint8(self):
        paddle.disable_static(paddle.CustomPlace("intel_gpu", 0))
        input_1 = np.random.random([5, 1, 10]).astype("uint8")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        self.assertTrue(np.allclose(expected_out, out_np))

    def test_out_float32(self):
        paddle.disable_static(paddle.CustomPlace("intel_gpu", 0))
        input_1 = np.random.random([5, 1, 10]).astype("float32")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        self.assertTrue(np.allclose(expected_out, out_np))


class TestDygraphReshapeInplaceAPI(TestDygraphReshapeAPI):
    def executed_api(self):
        self.reshape = paddle.reshape_


if __name__ == "__main__":
    unittest.main()
