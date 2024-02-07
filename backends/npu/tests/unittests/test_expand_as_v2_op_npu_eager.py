# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle

from npu_utils import check_soc_version

np.random.seed(10)


class TestExpandAsV2OpBf16(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "expand_as_v2"
        x = convert_float_to_uint16(np.random.rand(100).astype("float32"))
        target_tensor = np.random.rand(2, 100).astype("float32")
        self.inputs = {"X": x}
        self.attrs = {"target_shape": target_tensor.shape}
        bcast_dims = [2, 1]
        output = np.tile(convert_uint16_to_float(self.inputs["X"]), bcast_dims)
        self.outputs = {"Out": output}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)


class TestExpandAsV2OpFp16(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "expand_as_v2"
        x = np.random.rand(100).astype("float16")
        target_tensor = np.random.rand(2, 100).astype("float16")
        self.inputs = {"X": x}
        self.attrs = {"target_shape": target_tensor.shape}
        bcast_dims = [2, 1]
        output = np.tile(self.inputs["X"], bcast_dims)
        self.outputs = {"Out": output}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)


if __name__ == "__main__":
    unittest.main()
