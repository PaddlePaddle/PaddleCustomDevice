#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest

from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2022


class TestIdentity(OpTest):
    def set_sdaa(self):
        import os

        os.environ["HIGH_PERFORMANCE_CONV"] = "1"
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.__class__.no_need_check_grad = False
        self.__class__.op_type = "npu_identity"
        self.init_test_case()

    def init_test_case(self):
        self.input_size = [64, 224, 224, 3]  # NHWC

    def test_api_dygraph(self):
        import paddle.nn as nn

        paddle.set_device("sdaa")
        paddle.disable_static()
        conv_cpu = nn.Conv2D(3, 64, (7, 7), data_format="NHWC")
        conv_weight_numpy = conv_cpu.weight.numpy()
        x = np.random.uniform(low=0, high=1.0, size=self.input_size)
        x_var_cpu = paddle.to_tensor(x, dtype="float32")
        y = conv_cpu(x_var_cpu)
        conv_weight_numpy_storage = conv_cpu.weight.numpy()
        np.testing.assert_allclose(
            conv_weight_numpy, conv_weight_numpy_storage, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
