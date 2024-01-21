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
import numpy as np
import paddle

from npu_utils import check_soc_version


class TestAssignBF16(unittest.TestCase):
    def setUp(self):
        self.shape_x = [1, 3, 2, 4]
        self.x = np.random.uniform(0.1, 1, self.shape_x).astype(np.float32)

    @check_soc_version
    def test_assign_bf16(self):
        npu_x = paddle.to_tensor(self.x, "bfloat16")
        npu_out = paddle.assign(npu_x)
        np.testing.assert_allclose(
            npu_x.cast("float32").numpy(), npu_out.cast("float32").numpy(), rtol=1e-08
        )


if __name__ == "__main__":
    unittest.main()
