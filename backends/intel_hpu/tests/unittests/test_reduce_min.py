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
import numpy as np
from tests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.base.core as core

paddle.enable_static()


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestNPUReduceMinOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [-1]}
        self.outputs = {"Out": self.inputs["X"].min(axis=tuple(self.attrs["dim"]))}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def init_dtype(self):
        self.dtype = np.float32


# @skip_check_grad_ci(
#     reason="reduce_min is discontinuous non-derivable function,"
#     " its gradient check is not supported by unittest framework."
# )
# class TestReduceMinOpMultiAxises(TestNPUReduceMinOp):
#     """Remove Min with subgradient from gradient check to confirm the success of CI."""

#     def setUp(self):
#         self.op_type = "reduce_min"
#         self.set_npu()
#         self.init_dtype()

#         self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
#         self.attrs = {"dim": [-2, -1]}
#         self.outputs = {"Out": self.inputs["X"].min(axis=tuple(self.attrs["dim"]))}


# @skip_check_grad_ci(
#     reason="reduce_min is discontinuous non-derivable function,"
#     " its gradient check is not supported by unittest framework."
# )
# class TestReduceAll(TestNPUReduceMinOp):
#     """Remove Min with subgradient from gradient check to confirm the success of CI."""

#     def setUp(self):
#         self.op_type = "reduce_min"
#         self.set_npu()
#         self.init_dtype()

#         self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
#         self.attrs = {"reduce_all": True}
#         self.outputs = {"Out": self.inputs["X"].min()}


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMinOpWithOutDtype_fp16(TestNPUReduceMinOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [-2], "out_dtype": int(core.VarDesc.VarType.FP16)}
        self.outputs = {
            "Out": self.inputs["X"]
            .min(axis=tuple(self.attrs["dim"]))
            .astype(np.float16)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMinOpWithOutDtype_fp32(TestNPUReduceMinOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [-2], "out_dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {
            "Out": self.inputs["X"]
            .min(axis=tuple(self.attrs["dim"]))
            .astype(np.float32)
        }




@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMinOpWithOutDtype_fp32_2(TestNPUReduceMinOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [-2], "out_dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {
            "Out": self.inputs["X"]
            .min(axis=tuple(self.attrs["dim"]))
            .astype(np.float32)
        }

    def init_dtype(self):
        self.dtype = np.float16


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceMinOpInt64(TestNPUReduceMinOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [-1], "out_dtype": int(core.VarDesc.VarType.INT64)}
        self.outputs = {
            "Out": self.inputs["X"]
            .min(axis=tuple(self.attrs["dim"]))
            .astype(np.float32)
        }

    def init_dtype(self):
        self.dtype = np.int64


if __name__ == "__main__":
    unittest.main()
