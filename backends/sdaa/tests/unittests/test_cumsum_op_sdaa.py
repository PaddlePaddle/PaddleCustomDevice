#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np

from op_test import OpTest
import paddle
import paddle.base.core as core
import paddle.base as base

paddle.enable_static()


class TestCumsumOp(unittest.TestCase):
    def run_cases(self):
        data_np = np.arange(12).reshape(3, 4).astype("float32")
        data = paddle.to_tensor(data_np)

        y = paddle.cumsum(data)
        z = np.cumsum(data_np)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumsum(data, axis=0)
        z = np.cumsum(data_np, axis=0)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumsum(data, axis=-1)
        z = np.cumsum(data_np, axis=-1)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumsum(data, dtype="float32")
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP32)

        y = paddle.cumsum(data, axis=-2)
        z = np.cumsum(data_np, axis=-2)
        self.assertTrue(np.array_equal(z, y.numpy()))

    def run_static(self, use_custom_device=False):
        with base.program_guard(base.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data("X", [100, 100])
            y = paddle.cumsum(x)
            y2 = paddle.cumsum(x, axis=0)
            y3 = paddle.cumsum(x, axis=-1)
            y4 = paddle.cumsum(x, dtype="float32")
            y5 = paddle.cumsum(x, dtype=np.float16)
            y6 = paddle.cumsum(x, axis=-2)

            place = (
                base.CustomPlace("sdaa", 0) if use_custom_device else base.CPUPlace()
            )
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            out = exe.run(
                feed={"X": data_np},
                fetch_list=[y.name, y2.name, y3.name, y4.name, y5.name, y6.name],
            )

            z = np.cumsum(data_np)
            self.assertTrue(np.allclose(z, out[0]))
            z = np.cumsum(data_np, axis=0)
            self.assertTrue(np.allclose(z, out[1]))
            z = np.cumsum(data_np, axis=-1)
            self.assertTrue(np.allclose(z, out[2]))
            self.assertTrue(out[3].dtype == np.float32)
            self.assertTrue(out[4].dtype == np.float16)
            z = np.cumsum(data_np, axis=-2)
            self.assertTrue(np.allclose(z, out[5]))

    def test_npu(self):
        # Now, sdaa tests need setting paddle.enable_static()

        self.run_static(use_custom_device=True)

    def test_name(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data("x", [3, 4])
            y = paddle.cumsum(x, name="out")
            self.assertTrue("out" in y.name)


def cumsum_wrapper(x, axis=-1, flatten=False, exclusive=False, reverse=False):
    return paddle._C_ops.cumsum(x, axis, flatten, exclusive, reverse)


class TestCumSumOp1(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.python_api = cumsum_wrapper
        self.public_python_api = paddle.cumsum
        self.set_sdaa()
        self.init_dtype()
        self.init_testcase()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_testcase(self):
        self.attrs = {"axis": 2}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=2)}


class TestCumSumOp2(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": -1, "reverse": True}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {
            "Out": np.flip(np.flip(self.inputs["X"], axis=2).cumsum(axis=2), axis=2)
        }


class TestCumSumOp3(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 1}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=1)}


class TestCumSumOp4(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 0}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=0)}


class TestCumSumOp5(TestCumSumOp1):
    def init_testcase(self):
        self.inputs = {"X": np.random.random((5, 20)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=1)}


class TestCumSumOp7(TestCumSumOp1):
    def init_testcase(self):
        self.inputs = {"X": np.random.random((100)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=0)}


class TestCumSumExclusive1(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((4, 5, 65)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2,
            )
        }


class TestCumSumExclusive2(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((1, 1, 888)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2,
            )
        }


class TestCumSumExclusive3(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((4, 5, 888)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2,
            )
        }


class TestCumSumExclusive4(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((1, 1, 3049)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2,
            )
        }


class TestCumSumExclusive5(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((4, 5, 3096)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2,
            )
        }


class TestCumSumReverseExclusive(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "reverse": True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype(self.dtype)
        self.inputs = {"X": a}
        a = np.flip(a, axis=2)
        self.outputs = {
            "Out": np.concatenate(
                (
                    np.flip(a[:, :, :-1].cumsum(axis=2), axis=2),
                    np.zeros((4, 5, 1), dtype=self.dtype),
                ),
                axis=2,
            )
        }


class TestCumSumWithFlatten1(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"flatten": True}
        self.inputs = {"X": np.random.random((5, 6)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumsum()}


class TestCumSumWithFlatten2(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"flatten": True}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumsum()}


# ----------------Cumsum Int64----------------
class TestCumSumOpInt64(TestCumSumOp1):
    def init_testcase(self):
        self.attrs = {"axis": -1, "reverse": True}
        self.inputs = {
            "X": np.random.randint(1, 10000, size=(5, 6, 10)).astype(self.dtype)
        }
        self.outputs = {
            "Out": np.flip(np.flip(self.inputs["X"], axis=2).cumsum(axis=2), axis=2)
        }


def create_test_int64(parent):
    class TestCumSumInt64(parent):
        def init_dtype(self):
            self.dtype = np.int64

    cls_name = "{0}_{1}".format(parent.__name__, "Int64")
    TestCumSumInt64.__name__ = cls_name
    globals()[cls_name] = TestCumSumInt64


create_test_int64(TestCumSumOp1)
create_test_int64(TestCumSumOp3)
create_test_int64(TestCumSumOp4)
create_test_int64(TestCumSumOp5)
create_test_int64(TestCumSumOp7)
create_test_int64(TestCumSumWithFlatten1)
create_test_int64(TestCumSumWithFlatten2)
# tecodnn only support reverse=false, exclusive=false when dtype is int64
# create_test_int64(TestCumSumOp2)
# create_test_int64(TestCumSumExclusive1)
# create_test_int64(TestCumSumExclusive2)
# create_test_int64(TestCumSumExclusive3)
# create_test_int64(TestCumSumExclusive4)
# create_test_int64(TestCumSumExclusive5)
# create_test_int64(TestCumSumReverseExclusive)

if __name__ == "__main__":
    unittest.main()
