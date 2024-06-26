# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function

import numpy as np
import unittest

import paddle
import paddle.base as base
from paddle.base import Program, program_guard

paddle.enable_static()
SEED = 2021


class TestMemcpy_FillConstant(unittest.TestCase):
    def get_prog(self):
        self.__class__.use_custom_device = True
        paddle.enable_static()
        main_program = Program()
        with program_guard(main_program):
            cpu_var_name = "tensor@Cpu"
            sdaa_var_name = "tensor@sdaa"
            cpu_var = main_program.global_block().create_var(
                name=cpu_var_name,
                shape=[10, 10],
                dtype="float32",
                persistable=False,
                stop_gradient=True,
            )
            sdaa_var = main_program.global_block().create_var(
                name=sdaa_var_name,
                shape=[10, 10],
                dtype="float32",
                persistable=False,
                stop_gradient=True,
            )
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": sdaa_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": sdaa_var.dtype,
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
        return main_program, sdaa_var, cpu_var

    def test_sdaa_cpoy_to_cpu(self):
        self.__class__.use_custom_device = True
        main_program, sdaa_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(
            type="memcpy",
            inputs={"X": sdaa_var},
            outputs={"Out": cpu_var},
            attrs={"dst_place_type": 0},
        )
        place = paddle.CustomPlace("sdaa", 0)
        exe = base.Executor(place)
        sdaa_, cpu_ = exe.run(
            main_program, feed={}, fetch_list=[sdaa_var.name, cpu_var.name]
        )
        np.testing.assert_allclose(sdaa_, cpu_)
        np.testing.assert_allclose(cpu_, np.ones((10, 10)))

    def test_cpu_cpoy_sdaa(self):
        self.__class__.use_custom_device = True
        main_program, sdaa_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(
            type="memcpy",
            inputs={"X": cpu_var},
            outputs={"Out": sdaa_var},
            attrs={"dst_place_type": 6},
        )
        place = paddle.CustomPlace("sdaa", 0)
        exe = base.Executor(place)
        sdaa_, cpu_ = exe.run(
            main_program, feed={}, fetch_list=[sdaa_var.name, cpu_var.name]
        )
        np.testing.assert_allclose(sdaa_, cpu_)
        np.testing.assert_allclose(sdaa_, np.zeros((10, 10)))


if __name__ == "__main__":
    unittest.main()
