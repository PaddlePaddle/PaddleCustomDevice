# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle
import paddle.base.dygraph as dg


class TestAbs(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32"]
        self._places = [paddle.CustomPlace("iluvatar_gpu", 0)]

    def test_all_positive(self):
        for dtype in self._dtypes:
            x = 1 + 10 * np.random.random([13, 3, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    y = paddle.abs(paddle.to_tensor(x))
                    np.testing.assert_allclose(np.abs(x), y.numpy(), rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
