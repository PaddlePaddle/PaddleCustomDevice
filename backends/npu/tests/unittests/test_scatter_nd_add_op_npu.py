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
from tests.op_test import OpTest

import paddle
from paddle import fluid
from paddle.fluid.dygraph.base import switch_to_static_graph


def numpy_scatter_nd(ref, index, updates, fun):
    ref_shape = ref.shape
    index_shape = index.shape

    end_size = index_shape[-1]
    remain_numl = 1
    for i in range(len(index_shape) - 1):
        remain_numl *= index_shape[i]

    slice_size = 1
    for i in range(end_size, len(ref_shape)):
        slice_size *= ref_shape[i]

    flat_index = index.reshape([remain_numl] + list(index_shape[-1:]))
    flat_updates = updates.reshape((remain_numl, slice_size))
    flat_output = ref.reshape(list(ref_shape[:end_size]) + [slice_size])

    for i_up, i_out in enumerate(flat_index):
        i_out = tuple(i_out)
        flat_output[i_out] = fun(flat_output[i_out], flat_updates[i_up])
    return flat_output.reshape(ref.shape)


def numpy_scatter_nd_add(ref, index, updates):
    return numpy_scatter_nd(ref, index, updates, lambda x, y: x + y)


def judge_update_shape(ref, index):
    ref_shape = ref.shape
    index_shape = index.shape
    update_shape = []
    for i in range(len(index_shape) - 1):
        update_shape.append(index_shape[i])
    for i in range(index_shape[-1], len(ref_shape), 1):
        update_shape.append(ref_shape[i])
    return update_shape


class TestScatterNdAddSimpleOp(OpTest):
    """
    A simple example
    """

    def setUp(self):
        self.set_npu()
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        self.public_python_api = paddle.scatter_nd_add
        self._set_dtype()
        if self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        ref_np = np.random.random([100]).astype(target_dtype)
        index_np = np.random.randint(0, 100, [100, 1]).astype("int32")
        updates_np = np.random.random([100]).astype(target_dtype)
        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {"X": ref_np, "Index": index_np, "Updates": updates_np}
        self.outputs = {"Out": expect_np}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X", "Updates"], "Out")


class TestScatterNdAddSimpleFP16Op(TestScatterNdAddSimpleOp):
    """
    A simple example
    """

    def _set_dtype(self):
        self.dtype = np.float16


class TestScatterNdAddWithEmptyIndex(OpTest):
    """
    Index has empty element
    """

    def setUp(self):
        self.set_npu()
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        self.public_python_api = paddle.scatter_nd_add

        self._set_dtype()
        if self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        ref_np = np.random.random((10, 10)).astype(target_dtype)
        index_np = np.array([[], []]).astype("int32")
        updates_np = np.random.random((2, 10, 10)).astype(target_dtype)

        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {"X": ref_np, "Index": index_np, "Updates": updates_np}
        self.outputs = {"Out": expect_np}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X", "Updates"], "Out")


class TestScatterNdAddWithEmptyIndexFP16(TestScatterNdAddWithEmptyIndex):
    """
    Index has empty element
    """

    def _set_dtype(self):
        self.dtype = np.float16


class TestScatterNdAddWithHighRankSame(OpTest):
    """
    Both Index and X have high rank, and Rank(Index) = Rank(X)
    """

    def setUp(self):
        self.set_npu()
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        self.public_python_api = paddle.scatter_nd_add

        if self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        shape = (3, 2, 2, 1, 10)
        ref_np = np.random.rand(*shape).astype(target_dtype)
        index_np = np.vstack(
            [np.random.randint(0, s, size=100) for s in shape]
        ).T.astype("int32")
        update_shape = judge_update_shape(ref_np, index_np)
        updates_np = np.random.rand(*update_shape).astype(target_dtype)
        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {"X": ref_np, "Index": index_np, "Updates": updates_np}
        self.outputs = {"Out": expect_np}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X", "Updates"], "Out")


class TestScatterNdAddWithHighRankSameFP16(TestScatterNdAddWithHighRankSame):
    """
    Both Index and X have high rank, and Rank(Index) = Rank(X)
    """

    def _set_dtype(self):
        self.dtype = np.float16


# Test Python API
class TestScatterNdOpAPI(unittest.TestCase):
    """
    test scatter_nd_add api and scatter_nd api
    """

    def testcase5(self):
        shape = [2, 3, 4]
        x = np.arange(int(np.prod(shape))).reshape(shape)
        index = np.array([[0, 0, 2], [0, 1, 2]])
        val = np.array([-1, -3])

        with fluid.dygraph.guard():
            paddle.set_device("npu")
            npu_value = paddle.scatter_nd_add(
                paddle.to_tensor(x),
                paddle.to_tensor(index),
                paddle.to_tensor(val),
            )
            paddle.set_device("cpu")
            cpu_value = paddle.scatter_nd_add(
                paddle.to_tensor(x),
                paddle.to_tensor(index),
                paddle.to_tensor(val),
            )
            np.testing.assert_array_equal(npu_value.numpy(), cpu_value.numpy())
            paddle.set_device("npu")

        @switch_to_static_graph
        def test_static_graph():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_t = paddle.static.data(name="x", dtype=x.dtype, shape=x.shape)
                index_t = paddle.static.data(
                    name="index", dtype=index.dtype, shape=index.shape
                )
                val_t = paddle.static.data(name="val", dtype=val.dtype, shape=val.shape)
                out_t = paddle.scatter_nd_add(x_t, index_t, val_t)
                feed = {x_t.name: x, index_t.name: index, val_t.name: val}
                fetch = [out_t]

                npu_exe = paddle.static.Executor(paddle.CustomPlace("npu", 0))
                gpu_value = npu_exe.run(feed=feed, fetch_list=fetch)[0]
                cpu_exe = paddle.static.Executor(paddle.CPUPlace())
                cpu_value = cpu_exe.run(feed=feed, fetch_list=fetch)[0]
                np.testing.assert_array_equal(gpu_value, cpu_value)

        test_static_graph()


class TestDygraph(unittest.TestCase):
    def test_dygraph(self):
        with fluid.dygraph.guard(paddle.CustomPlace("npu", 0)):
            x = paddle.rand(shape=[3, 5, 9, 10], dtype="float32")
            updates = paddle.rand(shape=[3, 9, 10], dtype="float32")
            index_data = np.array([[1, 1], [0, 1], [1, 3]]).astype(np.int64)
            index = fluid.dygraph.to_variable(index_data)
            output = paddle.scatter_nd_add(x, index, updates)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
