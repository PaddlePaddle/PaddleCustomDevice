# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle_custom_device


def get_cann_version():
    return int(paddle_custom_device.npu.version()["cann"].split(".")[0])


def check_soc_version(func):
    def wrapper(self):
        version_code = get_cann_version()
        if version_code >= 7:
            return func(self)
        self.__class__.op_type = "skip"
        return unittest.skip("Skipping the test case since cann is not 7.x")(func)(self)

    return wrapper
