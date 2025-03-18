#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.#
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
from tests.op_test import OpTest
import paddle
import paddle.base.core as core

paddle.enable_static()

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


class TestOneHotOp(OpTest):
    def set_hpu(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_hpu()
        self.op_type = "one_hot_v2"
        self.dtype = np.int32
        depth = 10
        depth_np = np.array(10).astype("int32")
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.prod(x.shape), depth)).astype(self.dtype)

        for i in range(np.prod(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {"X": (x, x_lod), "depth_tensor": depth_np}
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        )

    def test_check_grad(self):
        pass


class TestOneHotOpAttr(OpTest):
    def set_hpu(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_hpu()
        self.op_type = "one_hot_v2"
        self.dtype = np.int32
        depth = 10
        depth_np = np.array(10).astype("int32")
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.prod(x.shape), depth)).astype(self.dtype)

        for i in range(np.prod(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {"X": (x, x_lod)}
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32), "depth": depth}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        )

    def test_check_grad(self):
        pass


class TestOneHotOpNoLod(OpTest):
    def set_hpu(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_hpu()
        self.op_type = "one_hot_v2"
        self.dtype = np.int32
        depth = 10
        depth_np = np.array(10).astype("int32")
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.prod(x.shape), depth)).astype(self.dtype)

        for i in range(np.prod(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {"X": x, "depth_tensor": depth_np}
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        )

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
