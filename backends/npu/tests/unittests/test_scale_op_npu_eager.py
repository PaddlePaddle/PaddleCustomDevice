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

import numpy as np
import unittest

from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle

from npu_utils import check_soc_version


class Testbf16Scale(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "scale"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_dtype()

        middle_inputs = np.random.random((10, 10)).astype(self.dtype)
        middle_inputs = convert_float_to_uint16(middle_inputs)
        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(middle_inputs)}
        self.attrs = {"scale": -2.3, "bias": 0, "bias_after_scale": True}
        self.outputs = {
            "Out": (
                convert_uint16_to_float(middle_inputs) * self.dtype(self.attrs["scale"])
            ).astype(self.dtype)
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
