# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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
import paddle


class TestDevice(unittest.TestCase):
    def test_device(self):
        dt = paddle.device.get_all_custom_device_type()
        self.assertTrue(len(dt), 1)
        self.assertTrue(dt[0], "musa")

        device = paddle.get_device()
        self.assertTrue(device, "musa:0")

        paddle.set_device("musa:1")
        device = paddle.get_device()
        self.assertTrue(device, "musa:1")
