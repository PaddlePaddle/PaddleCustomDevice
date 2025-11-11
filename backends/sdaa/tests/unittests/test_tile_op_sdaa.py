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

from op_test import OpTest, skip_check_grad_ci
import paddle
from paddle import base
from paddle.base import Program, program_guard

paddle.enable_static()
SEED = 1024


# Situation 1: repeat_times is a list (without tensor)
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileOpRank1(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.public_python_api = paddle.tile
        self.init_data()

        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"repeat_times": self.repeat_times}
        output = np.tile(self.inputs["X"], self.repeat_times)
        self.outputs = {"Out": output}

    def init_data(self):
        self.ori_shape = [100]
        self.repeat_times = [2]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTileOpRank_ZeroDim1(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = []
        self.repeat_times = []


class TestTileOpRank_ZeroDim2(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = []
        self.repeat_times = [2]


class TestTileOpRank_ZeroDim3(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = []
        self.repeat_times = [2, 3]


# with dimension expanding
class TestTileOpRank2Expanding(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = [120]
        self.repeat_times = [2, 2]


class TestTileOpRank2(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]


class TestTileOpRank3_Corner(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.repeat_times = (1, 1, 1)


# failed
class TestTileOpRank3_Corner2(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.repeat_times = (2, 2)


class TestTileOpRank3(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 15)
        self.repeat_times = (2, 1, 4)


class TestTileOpRank4(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.repeat_times = (3, 2, 1, 2)


# Situation 2: repeat_times is a list (with tensor)
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileOpRank1_tensor_attr(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_data()
        repeat_times_tensor = []
        for index, ele in enumerate(self.repeat_times):
            repeat_times_tensor.append(
                ("x" + str(index), np.ones(1).astype("int32") * ele)
            )

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "repeat_times_tensor": repeat_times_tensor,
        }
        self.attrs = {"repeat_times": self.infer_repeat_times}
        output = np.tile(self.inputs["X"], self.repeat_times)
        self.outputs = {"Out": output}

    def init_data(self):
        self.ori_shape = [100]
        self.repeat_times = [2]
        self.infer_repeat_times = [-1]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTileOpRank2_Corner_tensor_attr(TestTileOpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.repeat_times = [1, 1]
        self.infer_repeat_times = [1, -1]


class TestTileOpRank2_attr_tensor(TestTileOpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]
        self.infer_repeat_times = [-1, 3]


# Situation 3: repeat_times is a tensor
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileOpRank1_tensor(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_data()

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            "RepeatTimes": np.array(self.repeat_times).astype("int32"),
        }
        self.attrs = {}
        output = np.tile(self.inputs["X"], self.repeat_times)
        self.outputs = {"Out": output}

    def init_data(self):
        self.ori_shape = [100]
        self.repeat_times = [2]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTileOpRank2_tensor(TestTileOpRank1_tensor):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]


# Situation 4: input x is Double
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileOpDouble(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.inputs = {"X": np.random.randint(10, size=(4, 4, 5)).astype("double")}
        self.attrs = {"repeat_times": [2, 1, 4]}
        output = np.tile(self.inputs["X"], (2, 1, 4))
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output_with_place(self.place)


# Situation 5: input x is Fp16
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileFP16OP(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.dtype = np.float16
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.enable_cinn = True
        self.public_python_api = paddle.tile
        self.init_data()
        x = np.random.uniform(10, size=self.ori_shape).astype(self.dtype)
        output = np.tile(x, self.repeat_times)
        self.inputs = {"X": x}
        self.attrs = {"repeat_times": self.repeat_times}
        self.outputs = {"Out": output}

    def init_data(self):
        self.dtype = np.float16
        self.ori_shape = [100, 4, 5]
        self.repeat_times = [2, 1, 4]

    def test_check_output(self):
        self.check_output_with_place(self.place)


# Situation 6: input x is Bool
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileOpBoolean(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.inputs = {"X": np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {"repeat_times": [2, 1, 4]}
        output = np.tile(self.inputs["X"], (2, 1, 4))
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output_with_place(self.place)


# Situation 7: input x is Int32
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileOpInteger(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.inputs = {"X": np.random.randint(10, size=(4, 4, 5)).astype("int32")}
        self.attrs = {"repeat_times": [2, 1, 4]}
        output = np.tile(self.inputs["X"], (2, 1, 4))
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output_with_place(self.place)


# Situation 8: input x is Int64
@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestTileOpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "tile"
        self.python_api = paddle.tile
        self.place = paddle.CustomPlace("sdaa", 0)
        self.inputs = {"X": np.random.randint(10, size=(2, 4, 5)).astype("int64")}
        self.attrs = {"repeat_times": [2, 1, 4]}
        output = np.tile(self.inputs["X"], (2, 1, 4))
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTileError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], paddle.CustomPlace("sdaa", 0)
            )
            repeat_times = [2, 2]
            self.assertRaises(TypeError, paddle.tile, x1, repeat_times)
            x2 = paddle.static.data(name="x2", shape=[-1, 4], dtype="uint8")
            self.assertRaises(TypeError, paddle.tile, x2, repeat_times)
            x3 = paddle.static.data(name="x3", shape=[-1, 4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tile, x3, repeat_times)


class TestTileAPIStatic(unittest.TestCase):
    def test_api(self):
        paddle.device.set_device("sdaa")
        with program_guard(Program(), Program()):
            repeat_times = [2, 2]
            x1 = paddle.static.data(name="x1", shape=[-1, 4], dtype="int32")
            out = paddle.tile(x1, repeat_times)
            positive_2 = paddle.tensor.fill_constant([1], dtype="int32", value=2)
            out2 = paddle.tile(x1, repeat_times=[positive_2, 2])


# Test python API
class TestTileAPI(unittest.TestCase):
    def test_api(self):
        paddle.device.set_device("sdaa")
        with base.dygraph.guard():
            np_x = np.random.random([12, 14]).astype("float32")
            x = paddle.to_tensor(np_x)

            positive_2 = np.array([2]).astype("int32")
            positive_2 = paddle.to_tensor(positive_2)

            repeat_times = np.array([2, 3]).astype("int32")
            repeat_times = paddle.to_tensor(repeat_times)

            out_1 = paddle.tile(x, repeat_times=[2, 3])
            out_2 = paddle.tile(x, repeat_times=[positive_2, 3])
            out_3 = paddle.tile(x, repeat_times=repeat_times)

            assert np.array_equal(out_1.numpy(), np.tile(np_x, (2, 3)))
            assert np.array_equal(out_2.numpy(), np.tile(np_x, (2, 3)))
            assert np.array_equal(out_3.numpy(), np.tile(np_x, (2, 3)))


class TestTileAPI_ZeroDim(unittest.TestCase):
    def test_dygraph(self):
        paddle.device.set_device("sdaa")
        paddle.disable_static()

        x = paddle.rand([])
        x.stop_gradient = False

        out = paddle.tile(x, [])
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.grad.shape, [])

        out = paddle.tile(x, [3])
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [3])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.grad.shape, [3])

        out = paddle.tile(x, [2, 3])
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [2, 3])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.grad.shape, [2, 3])

        paddle.enable_static()


class TestFp16TileOp(unittest.TestCase):
    def testFp16(self):
        input_x = (np.random.random([1, 2, 3])).astype("float16")
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[1, 2, 3], dtype="float16")
            repeat_times = [2, 2]
            out = paddle.tile(x, repeat_times=repeat_times)
            if paddle.is_compiled_with_cuda():
                place = paddle.CustomPlace("sdaa", 0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                out = exe.run(feed={"x": input_x}, fetch_list=[out])


if __name__ == "__main__":
    unittest.main()
