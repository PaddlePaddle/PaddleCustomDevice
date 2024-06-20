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

from op_test import OpTest
import paddle
import paddle.base as base
import paddle.base.core as core

paddle.enable_static()


def one_hot_wrapper(x, depth_tensor, **keargs):
    return paddle.nn.functional.one_hot(x, depth_tensor)


class TestOneHotOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.op_type = "one_hot_v2"
        self.python_api = one_hot_wrapper
        self.python_out_sig = ["Out"]
        depth = 10
        depth_np = np.array(10).astype("int32")
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.product(x.shape), depth)).astype("float32")

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {"X": (x, x_lod), "depth_tensor": depth_np}
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            check_dygraph=False,
        )


class TestOneHotOp_non_lod(OpTest):
    def setUp(self):
        self.op_type = "one_hot_v2"
        self.python_api = one_hot_wrapper
        self.python_out_sig = ["Out"]
        depth = 10
        depth_np = np.array(10).astype("int32")
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.product(x.shape), depth)).astype("float32")

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {"X": x, "depth_tensor": depth_np}
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            check_dygraph=False,
        )


class TestOneHotOp_attr(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.op_type = "one_hot_v2"
        self.python_api = one_hot_wrapper
        self.python_out_sig = ["Out"]
        depth = 10
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]), 1, depth)).astype("float32")

        for i in range(np.product(x.shape)):
            out[i, 0, x[i]] = 1.0

        self.inputs = {"X": (x, x_lod)}
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32), "depth": depth}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            check_dygraph=False,
        )


class TestOneHotOp_default_dtype(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.op_type = "one_hot_v2"
        self.python_api = one_hot_wrapper
        self.python_out_sig = ["Out"]
        depth = 10
        depth_np = np.array(10).astype("int32")
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.product(x.shape), depth)).astype("float32")

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {"X": (x, x_lod), "depth_tensor": depth_np}
        self.attrs = {}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            check_dygraph=False,
        )


class TestOneHotOp_default_dtype_attr(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.op_type = "one_hot_v2"
        self.python_api = one_hot_wrapper
        self.python_out_sig = ["Out"]
        depth = 10
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]), 1, depth)).astype("float32")

        for i in range(np.product(x.shape)):
            out[i, 0, x[i]] = 1.0

        self.inputs = {"X": (x, x_lod)}
        self.attrs = {"depth": depth}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            check_dygraph=False,
        )


@unittest.skip("tecodnnOneHot API unable to check invalid value")
class TestOneHotOp_out_of_range(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.op_type = "one_hot_v2"
        self.python_api = one_hot_wrapper
        self.python_out_sig = ["Out"]
        depth = 10
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.choice([-1, depth]) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int32").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.product(x.shape), depth)).astype("float32")

        self.inputs = {"X": (x, x_lod)}
        self.attrs = {"depth": depth, "allow_out_of_range": True}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            check_dygraph=False,
        )


class TestOneHotOp_dtype_int64(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.op_type = "one_hot_v2"
        self.python_api = one_hot_wrapper
        self.python_out_sig = ["Out"]
        depth = 10
        depth_np = np.array(10).astype("int32")
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype("int64").reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.product(x.shape), depth)).astype("float32")

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {"X": (x, x_lod), "depth_tensor": depth_np}
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {"Out": (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            check_dygraph=False,
        )


class TestOneHotOpApi(unittest.TestCase):
    def test_api(self):
        depth = 10
        self._run(depth)

    def test_api_with_depthTensor(self):
        depth = paddle.assign(np.array([10], dtype=np.int32))
        self._run(depth)

    def test_api_with_dygraph(self):
        depth = 10
        label = np.array([np.random.randint(0, depth - 1) for i in range(6)]).reshape(
            [6, 1]
        )
        with base.dygraph.guard(paddle.CustomPlace("sdaa", 0)):
            one_hot_label = paddle.nn.functional.one_hot(
                base.dygraph.to_variable(label), depth
            )
            one_hot_label = paddle.nn.functional.one_hot(paddle.to_tensor(label), depth)

    def _run(self, depth):
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")
        one_hot_label = paddle.nn.functional.one_hot(x=label, num_classes=depth)

        place = paddle.CustomPlace("sdaa", 0)
        label_data = np.array([np.random.randint(0, 10 - 1) for i in range(6)]).reshape(
            [6, 1]
        )

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        ret = exe.run(
            feed={
                "label": label_data,
            },
            fetch_list=[one_hot_label],
            return_numpy=False,
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
