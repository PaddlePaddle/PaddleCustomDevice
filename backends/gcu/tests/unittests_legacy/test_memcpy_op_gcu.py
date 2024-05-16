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

import numpy as np
import unittest

import paddle
import paddle.base as base
from paddle.base import Program, program_guard

paddle.enable_static()


class TestMemcpy_FillConstant(unittest.TestCase):
    def get_prog(self):
        paddle.enable_static()
        main_program = Program()
        with program_guard(main_program):
            cpu_var_name = "tensor@Cpu"
            gcu_var_name = "tensor@Gcu"
            cpu_var = main_program.global_block().create_var(
                name=cpu_var_name,
                shape=[10, 10],
                dtype="float32",
                persistable=False,
                stop_gradient=True,
            )
            gcu_var = main_program.global_block().create_var(
                name=gcu_var_name,
                shape=[10, 10],
                dtype="float32",
                persistable=False,
                stop_gradient=True,
            )
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": gcu_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": gcu_var.dtype,
                    "value": 1.0,
                },
            )
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": cpu_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": cpu_var.dtype,
                    "value": 0.0,
                    "place_type": 0,
                },
            )
        return main_program, gcu_var, cpu_var

    def test_gcu_copy_to_cpu(self):
        self.__class__.use_custom_device = True
        place = paddle.CustomPlace("gcu", 0)
        main_program, gcu_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(
            type="memcpy",
            inputs={"X": gcu_var},
            outputs={"Out": cpu_var},
            attrs={"dst_place_type": 0},
        )
        exe = base.Executor(place)
        gcu_, cpu_ = exe.run(
            main_program, feed={}, fetch_list=[gcu_var.name, cpu_var.name]
        )
        np.testing.assert_allclose(gcu_, cpu_)
        np.testing.assert_allclose(cpu_, np.ones((10, 10)))

    def test_cpu_copy_to_gcu(self):
        self.__class__.use_custom_device = True
        place = paddle.CustomPlace("gcu", 0)
        main_program, gcu_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(
            type="memcpy",
            inputs={"X": cpu_var},
            outputs={"Out": gcu_var},
            attrs={"dst_place_type": 6},
        )
        exe = base.Executor(place)
        gcu_, cpu_ = exe.run(
            main_program, feed={}, fetch_list=[gcu_var.name, cpu_var.name]
        )
        np.testing.assert_allclose(gcu_, cpu_)
        np.testing.assert_allclose(gcu_, np.zeros((10, 10)))


if __name__ == "__main__":
    unittest.main()
