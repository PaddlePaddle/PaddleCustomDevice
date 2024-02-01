# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

import numpy as np
import paddle
import paddle.base.core as core

from npu_utils import check_soc_version
from tests.op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
    skip_check_grad_ci,
)

SEED = 2021


@skip_check_grad_ci(reason="[skip NPU cast grad check] not implemented yet.")
class TestCastBF16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "cast"
        self.place = paddle.CustomPlace("npu", select_npu)

        ipt = np.random.random(size=[10, 10]) + 1
        x = convert_float_to_uint16(ipt.astype("float32"))
        self.inputs = {"X": x}
        self.outputs = {"Out": convert_uint16_to_float(x)}

        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.BF16),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
