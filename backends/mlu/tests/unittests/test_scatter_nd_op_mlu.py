#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
        self.op_type = "scatter_nd_add"
        self.place = paddle.set_device("mlu")
        self.python_api = paddle.scatter_nd_add
        self.set_mlu()
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

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out")


class TestScatterNdAddSimpleFP16Op(TestScatterNdAddSimpleOp):
    """
    A simple example
    """

    def _set_dtype(self):
        self.dtype = np.float16


class TestScatterNdAddWithHighRankSame(OpTest):
    """
    Both Index and X have high rank, and Rank(Index) = Rank(X)
    """

    def setUp(self):
        self.op_type = "scatter_nd_add"
        self.place = paddle.set_device("mlu")
        self.python_api = paddle.scatter_nd_add
        self._set_dtype()
        self.set_mlu()
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

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def _set_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out")


class TestScatterNdAddWithHighRankSameFP16(TestScatterNdAddWithHighRankSame):
    """
    Both Index and X have high rank, and Rank(Index) = Rank(X)
    """

    def _set_dtype(self):
        self.dtype = np.float16


class TestScatterNdAddWithHighRankDiff(OpTest):
    """
    Both Index and X have high rank, and Rank(Index) < Rank(X)
    """

    def setUp(self):
        self.op_type = "scatter_nd_add"
        self.python_api = paddle.scatter_nd_add
        self.place = paddle.set_device("mlu")
        self.set_mlu()
        shape = (8, 2, 2, 1, 10)
        ref_np = np.random.rand(*shape).astype("float32")
        index = np.vstack([np.random.randint(0, s, size=500) for s in shape]).T
        index_np = index.reshape([10, 5, 10, 5]).astype("int64")
        update_shape = judge_update_shape(ref_np, index_np)
        updates_np = np.random.rand(*update_shape).astype("float32")
        expect_np = numpy_scatter_nd_add(ref_np.copy(), index_np, updates_np)

        self.inputs = {"X": ref_np, "Index": index_np, "Updates": updates_np}
        self.outputs = {"Out": expect_np}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out")


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
