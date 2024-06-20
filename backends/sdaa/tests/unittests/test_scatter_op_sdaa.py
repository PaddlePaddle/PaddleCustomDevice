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
import unittest
import numpy as np
import paddle
import paddle.base as base
from op_test import OpTest

paddle.enable_static()


class TestScatterOp(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 50)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 50)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad(self):
        pass


class TestScatterOp0(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.attrs = {"overwrite": True}
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad(self):
        pass


class TestScatterOp1(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        zeros_np = np.zeros([2, 3]).astype("float32")
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        self.attrs = {"overwrite": False}
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad(self):
        pass


class TestScatterOp2(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad(self):
        pass


class TestScatterAPI(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CustomPlace("sdaa", 0)]
        self.__class__.use_custom_device = True
        self.executed_api()

    def executed_api(self):
        self.scatter = paddle.scatter

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(name="input", shape=[3, 2], dtype="float32")
            index = paddle.static.data(name="index", shape=[4], dtype="int64")
            updates = paddle.static.data(name="updates", shape=[4, 2], dtype="float32")
            result = self.scatter(input, index, updates, False)

            input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={
                    "input": input_data,
                    "index": index_data,
                    "updates": updates_data,
                },
                fetch_list=[result],
            )
            self.assertEqual(
                (fetches[0] == np.array([[3.0, 3.0], [6.0, 6.0], [1.0, 1.0]])).all(),
                True,
            )

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
                index_data = np.array([2, 1, 0, 1]).astype(np.int64)
                updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(
                    np.float32
                )

                x = base.dygraph.to_variable(x_data)
                index = base.dygraph.to_variable(index_data)
                updates = base.dygraph.to_variable(updates_data)

                output1 = self.scatter(x, index, updates, overwrite=False)
                self.assertEqual(
                    (
                        output1.numpy()
                        == np.array([[3.0, 3.0], [6.0, 6.0], [1.0, 1.0]])
                    ).all(),
                    True,
                )


class TestScatterOpFp16(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float16")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float16")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.attrs = {"overwrite": True}
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
