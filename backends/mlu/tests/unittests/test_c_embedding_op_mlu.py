#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()
np.random.seed(10)


def get_c_embedding(start, end, table, ids):
    index = ids.flatten()
    input_mask = (index < start) | (index >= end)
    masked_input = index - start
    masked_input[input_mask] = 0
    output = table[masked_input]
    output[input_mask] = 0.0
    return output


def c_embedding_wrapper(table, index, start_index=0, vocab_size=-1):
    return paddle._C_ops.c_embedding(table, index, start_index, vocab_size)


class TestCEmbeddingMLU(OpTest):
    def setUp(self):
        self.set_mlu()
        self.init_dtype()
        self.initcase()

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(self.ids_dtype)
        self.start_index = 10
        self.end_index = self.start_index + 17
        self.vocab_size = 34

        self.inputs = {"W": table, "Ids": ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)

        self.outputs = {"Out": np_out.reshape((2, 4, 64))}
        self.attrs = {
            "start_index": self.start_index,
            "vocab_size": self.vocab_size,
        }

    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("mlu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["W"], "Out")

    def init_dtype(self):
        self.dtype = "float32"
        self.ids_dtype = "int64"


class TestCEmbeddingOpBase(TestCEmbeddingMLU):
    def setUp(self):
        self.init_dtype()
        self.initcase()
        self.set_mlu()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["W"], "Out")

    def init_dtype(self):
        self.dtype = "float32"
        self.ids_dtype = "int64"

    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("mlu", 0)


class TestCEmbeddingOpFP32(TestCEmbeddingOpBase):
    def setUp(self):
        self.init_dtype()
        self.initcase()
        self.set_mlu()

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(self.ids_dtype)
        self.start_index = 10
        ids[0][1] = 12
        ids[0][2] = 12
        ids[1][2] = 12
        ids[1][3] = 12
        self.end_index = self.start_index + 17

        self.inputs = {"W": table, "Ids": ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {"Out": np_out.reshape((2, 4, 64))}
        self.attrs = {"start_index": self.start_index}

    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("mlu", 0)

    def init_dtype(self):
        self.dtype = "float32"
        self.ids_dtype = "int32"


if __name__ == "__main__":
    unittest.main()
