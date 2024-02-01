# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)
import numpy as np
from tests.op_test import (
    OpTest,
    skip_check_grad_ci,
    convert_float_to_uint16,
    convert_uint16_to_float,
)
import paddle
import paddle.base.core as core
from npu_utils import check_soc_version


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestNPUReduceMaxOpBF16(OpTest):
    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            "X": convert_float_to_uint16(np.random.random((5, 6, 10)).astype("float32"))
        }
        self.attrs = {"dim": [-1]}
        self.outputs = {
            "Out": convert_uint16_to_float(self.inputs["X"]).max(
                axis=tuple(self.attrs["dim"])
            )
        }

    def init_dtype(self):
        self.dtype = np.uint16

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", select_npu)

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMaxOpMultiAxisesBF16(TestNPUReduceMaxOpBF16):
    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            "X": convert_float_to_uint16(np.random.random((5, 6, 10)).astype("float32"))
        }
        self.attrs = {"dim": [-2, -1]}
        self.outputs = {
            "Out": convert_uint16_to_float(self.inputs["X"]).max(
                axis=tuple(self.attrs["dim"])
            )
        }

    def init_dtype(self):
        self.dtype = "bfloat16"


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceAllBF16(TestNPUReduceMaxOpBF16):
    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            "X": convert_float_to_uint16(np.random.random((5, 6, 10)).astype("float32"))
        }
        self.attrs = {"reduce_all": True}
        self.outputs = {"Out": convert_uint16_to_float(self.inputs["X"]).max()}

    def init_dtype(self):
        self.dtype = "bfloat16"


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMaxOpWithOutDtype_BF16(TestNPUReduceMaxOpBF16):
    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            "X": convert_float_to_uint16(np.random.random((5, 6, 10)).astype("float32"))
        }
        self.attrs = {"dim": [-2, -1], "out_dtype": int(core.VarDesc.VarType.BF16)}
        self.outputs = {
            "Out": convert_uint16_to_float(self.inputs["X"])
            .max(axis=tuple(self.attrs["dim"]))
            .astype(np.float16)
        }


if __name__ == "__main__":
    unittest.main()
