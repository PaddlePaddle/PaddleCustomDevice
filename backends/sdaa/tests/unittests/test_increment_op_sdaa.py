# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

import unittest

import numpy as np
import paddle
from op_test import OpTest, convert_float_to_uint16

paddle.enable_static()
SEED = 2021


class TestIncrement(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np.array([1]).astype(self.dtype)),
        }

        self.attrs = {"Step": 1}
        self.outputs = {"Out": np.array([2])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestIncrementBF16(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(
                convert_float_to_uint16(np.array([1]).astype(self.dtype))
            ),
        }
        self.pre_input_id = id(self.inputs["X"])

        self.attrs = {"Step": 1}
        self.outputs = {"Out": np.array([2])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestIncrementFP16(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np.array([1]).astype(self.dtype)),
        }
        self.pre_input_id = id(self.inputs["X"])

        self.attrs = {"Step": 1}
        self.outputs = {"Out": np.array([2])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestIncrementFP64(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np.array([1]).astype(self.dtype)),
        }

        self.attrs = {"Step": 2.5}
        self.outputs = {"Out": np.array([2])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestIncrementINT64(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np.array([1]).astype(self.dtype)),
        }
        self.pre_input_id = id(self.inputs["X"])

        self.attrs = {"Step": 1}
        self.outputs = {"Out": np.array([2])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.int64

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestIncrementINT32(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np.array([1]).astype(self.dtype)),
        }
        self.pre_input_id = id(self.inputs["X"])

        self.attrs = {"Step": 10}
        self.outputs = {"Out": np.array([2])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.int32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestIncrementInplace(unittest.TestCase):
    def test_sdaa(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.array([1]).astype("float32")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[1], dtype="float32")
            b = paddle.increment(a)

        place = paddle.CustomPlace("sdaa", 0)

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        b_value = exe.run(
            main_prog,
            feed={
                "a": a_np,
            },
            fetch_list=[b],
        )

        print("input a id is : {}".format(id(a)))
        print("input b id is : {}".format(id(b)))

        self.assertEqual(id(a), id(b))
        self.assertEqual(b_value[0], 2)


if __name__ == "__main__":
    unittest.main()
