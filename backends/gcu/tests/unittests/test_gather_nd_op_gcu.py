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
from tests.op_test import OpTest

import numpy as np
import paddle.base as base
import paddle

paddle.enable_static()


def gather_nd_grad(x, index):
    # for TestGatherNdOpWithLowIndex
    dout_shape = index.shape[:-1] + x.shape[index.shape[-1] :]
    numel = 1
    for i in dout_shape:
        numel = numel * i
    dout = np.full(dout_shape, 1.0 / numel)
    dx = np.full_like(x, 0)

    index = tuple(index.reshape(-1, index.shape[-1]).T)
    np.add.at(dx, index, dout)

    return dx


class TestGatherNdOpWithEmptyIndex(OpTest):
    # Index has empty element, which means copy entire tensor
    def setUp(self):
        self.op_type = "gather_nd"
        self.set_device()
        self.set_dtype()
        xnp = np.random.random((5, 20)).astype(self.dtype)
        self.inputs = {"X": xnp, "Index": np.array([[], []]).astype("int32")}
        self.outputs = {"Out": np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.__class__.no_need_check_grad = True
        else:
            self.check_grad_with_place(self.place, ["X"], "Out")


class TestGatherNdOpWithEmptyIndexFp16(TestGatherNdOpWithEmptyIndex):
    def set_dtype(self):
        self.dtype = np.float16


class TestGatherNdOpWithLowIndex(OpTest):
    # Index has low rank, X has high rank

    def setUp(self):
        self.op_type = "gather_nd"
        self.set_device()
        self.set_dtype()
        xnp = np.random.uniform(0, 100, (10, 10)).astype(self.dtype)
        index = np.array([[1], [2]]).astype("int64")

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": xnp[tuple(index.T)]}
        self.x_grad = gather_nd_grad(xnp, index)

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.__class__.no_need_check_grad = True
        else:
            self.check_grad_with_place(
                self.place, ["X"], "Out", user_defined_grads=[self.x_grad]
            )


class TestGatherNdOpWithLowIndexFp16(TestGatherNdOpWithLowIndex):
    def set_dtype(self):
        self.dtype = np.float16


class TestGatherNdOpWithSameIndexAsX(OpTest):
    # Index has same rank as X's rank

    def setUp(self):
        self.op_type = "gather_nd"
        self.set_device()
        self.set_dtype()
        xnp = np.random.uniform(0, 100, (10, 10)).astype(self.dtype)
        index = np.array([[1, 1], [2, 1]]).astype("int64")

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": xnp[tuple(index.T)]}  # [25, 22]

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.__class__.no_need_check_grad = True
        else:
            self.check_grad_with_place(self.place, ["X"], "Out")


class TestGatherNdOpWithSameIndexAsXFp16(TestGatherNdOpWithSameIndexAsX):
    def set_dtype(self):
        self.dtype = np.float16


# Test Python API
class TestGatherNdAPI2(unittest.TestCase):
    def test_imperative(self):
        paddle.disable_static()
        paddle.set_device("gcu")
        input_1 = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
        index_1 = np.array([[1]]).astype("int32")
        input = base.dygraph.to_variable(input_1)
        index = base.dygraph.to_variable(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([3, 4])
        np.testing.assert_allclose(output_np[0], expected_output, rtol=1e-6)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
