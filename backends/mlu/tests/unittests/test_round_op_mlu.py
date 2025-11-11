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

import numpy as np
from tests.op_test import OpTest, skip_check_grad_ci

import paddle

paddle.enable_static()
SEED = 2024


@skip_check_grad_ci(reason="no need for round.")
class TestRound(OpTest):
    def setUp(self):
        self.set_mlu()
        self.op_type = "round"
        self.place = paddle.CustomPlace("mlu", 0)
        self.check_dygraph = True

        self.init_dtype()
        self.init_shape()
        np.random.seed(SEED)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.round(x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    def set_mlu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(self.place, check_dygraph=check_dygraph)


@skip_check_grad_ci(reason="no need for round.")
class TestRoundHalf(TestRound):
    def init_dtype(self):
        self.dtype = np.float16


@skip_check_grad_ci(reason="no need for round.")
class TestRound_ZeroDim(TestRound):
    def init_shape(self):
        self.shape = []
