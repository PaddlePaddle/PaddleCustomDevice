# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

paddle.enable_static()
np.random.seed(10)


class TestNewIrPrint(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.enable_static()
        place = paddle.CustomPlace("mlu", 0)
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones([2, 2], dtype="float32")
                y = paddle.ones([2, 2], dtype="float32")

                z = x + y
                z = paddle.static.Print(z)

            out = exe.run(main_program, {}, fetch_list=[z.name])

        gold_res = np.ones([2, 2], dtype="float32") * 2

        np.testing.assert_array_equal(out[0], gold_res)


if __name__ == "__main__":
    unittest.main()
