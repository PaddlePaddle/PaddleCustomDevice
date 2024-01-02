#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

from tests.op_test import OpTest
import paddle
import paddle.base as base
import paddle.base.core as core

paddle.enable_static()


class TestOneHotOp(OpTest):
    def setUp(self):
        self.op_type = "one_hot_v2"
        self.set_device()
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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestOneHotOp_non_lod(OpTest):
    def setUp(self):
        self.op_type = "one_hot_v2"
        self.set_device()
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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestOneHotOp_attr(OpTest):
    def setUp(self):
        self.op_type = "one_hot_v2"
        self.set_device()
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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestOneHotOp_default_dtype(OpTest):
    def setUp(self):
        self.op_type = "one_hot_v2"
        self.set_device()
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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestOneHotOp_default_dtype_attr(OpTest):
    def setUp(self):
        self.op_type = "one_hot_v2"
        self.set_device()
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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestOneHotOp_dtype_int64(OpTest):
    def setUp(self):
        self.op_type = "one_hot_v2"
        self.set_device()
        depth = 10
        depth_np = np.array(10).astype("int64")
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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


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
        with base.dygraph.guard(paddle.CustomPlace("gcu", 0)):
            one_hot_label = paddle.nn.functional.one_hot(
                base.dygraph.to_variable(label), depth
            )
            one_hot_label = paddle.nn.functional.one_hot(paddle.to_tensor(label), depth)

    def _run(self, depth):
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")
        one_hot_label = paddle.nn.functional.one_hot(x=label, num_classes=depth)

        place = paddle.CustomPlace("gcu", 0)
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


class BadInputTestOnehotV2(unittest.TestCase):
    def test_error(self):
        paddle.set_device("gcu")
        with base.program_guard(base.Program()):

            def test_bad_x():
                label = paddle.static.data(
                    name="label",
                    shape=[-1, 4],
                    dtype="float32",
                )
                label.desc.set_need_check_feed(False)
                one_hot_label = paddle.nn.functional.one_hot(x=label, num_classes=4)

            self.assertRaises(TypeError, test_bad_x)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
