#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np

from tests.op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

from npu_utils import check_soc_version


class TestElementwiseAddBF16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_add"
        self.place = paddle.CustomPlace("npu", 0)
        self.shape_x = [5, 6, 2, 10]
        self.shape_y = [5, 1, 2, 10]
        self.x = np.random.uniform(0.1, 1, self.shape_x).astype(np.float32)
        self.y = np.random.uniform(0.1, 1, self.shape_y).astype(np.float32)
        np_uint16_x = convert_float_to_uint16(self.x)
        np_uint16_y = convert_float_to_uint16(self.y)
        np_uint16_to_fp32_x = convert_uint16_to_float(np_uint16_x)
        np_uint16_to_fp32_y = convert_uint16_to_float(np_uint16_y)
        np_out = np.add(np_uint16_to_fp32_x, np_uint16_to_fp32_y)
        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np_uint16_x),
            "Y": OpTest.np_dtype_to_base_dtype(np_uint16_y),
        }
        self.outputs = {"Out": np_out}

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output_bf16(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad_bf16(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
            max_relative_error=4e-3,
        )


if __name__ == "__main__":
    unittest.main()
