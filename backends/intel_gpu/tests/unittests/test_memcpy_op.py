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

import paddle
import paddle.base as base
from paddle.base import Program, program_guard

SEED = 2021


class TestMemcpy_API(unittest.TestCase):
    def init_config(self):
        self.dtype = "float32"
        self.shape = [10, 10]

    def get_prog(self):
        self.init_config()
        self.__class__.use_custom_device = True
        paddle.enable_static()
        main_program = Program()
        with program_guard(main_program):
            cpu_var_name = "tensor@Cpu"
            intel_gpu_var_name = "tensor@intel_gpu"
            cpu_var = main_program.global_block().create_var(
                name=cpu_var_name,
                shape=self.shape,
                dtype=self.dtype,
                persistable=False,
                stop_gradient=True,
            )
            intel_gpu_var = main_program.global_block().create_var(
                name=intel_gpu_var_name,
                shape=self.shape,
                dtype=self.dtype,
                persistable=False,
                stop_gradient=True,
            )
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": intel_gpu_var_name},
                attrs={
                    "shape": intel_gpu_var.shape,
                    "dtype": intel_gpu_var.dtype,
                    "value": 1.0,
                },
            )
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": cpu_var_name},
                attrs={
                    "shape": cpu_var.shape,
                    "dtype": cpu_var.dtype,
                    "value": 0.0,
                    "place_type": 0,
                },
            )
        return main_program, intel_gpu_var, cpu_var

    def test_intel_gpu_copy_to_cpu(self):
        self.__class__.use_custom_device = True
        main_program, intel_gpu_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(
            type="memcpy",
            inputs={"X": intel_gpu_var},
            outputs={"Out": cpu_var},
            attrs={"dst_place_type": 0},
        )
        place = paddle.CustomPlace("intel_gpu", 0)
        exe = base.Executor(place)
        intel_gpu_, cpu_ = exe.run(
            main_program, feed={}, fetch_list=[intel_gpu_var.name, cpu_var.name]
        )
        np.testing.assert_allclose(intel_gpu_, cpu_)
        np.testing.assert_allclose(cpu_, np.ones(self.shape, dtype=self.dtype))

    def test_cpu_copy_intel_gpu(self):
        self.__class__.use_custom_device = True
        main_program, intel_gpu_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(
            type="memcpy",
            inputs={"X": cpu_var},
            outputs={"Out": intel_gpu_var},
            attrs={"dst_place_type": 6},
        )
        place = paddle.CustomPlace("intel_gpu", 0)
        exe = base.Executor(place)
        intel_gpu_, cpu_ = exe.run(
            main_program, feed={}, fetch_list=[intel_gpu_var.name, cpu_var.name]
        )
        np.testing.assert_allclose(intel_gpu_, cpu_)
        np.testing.assert_allclose(intel_gpu_, np.zeros(self.shape, dtype=self.dtype))


class TestMemcpy_3D(TestMemcpy_API):
    def init_config(self):
        self.dtype = "float32"
        self.shape = [15, 10, 5]


if __name__ == "__main__":
    unittest.main()
