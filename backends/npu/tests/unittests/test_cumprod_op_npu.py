# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from tests.op_test import OpTest
import paddle
import paddle.base.core as core
import paddle.base as base

paddle.enable_static()


class TestCumprodOp(unittest.TestCase):
    def run_cases(self):
        data_np = np.arange(12).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.cumprod(data)
        z = np.cumprod(data_np)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumprod(data, axis=0)
        z = np.cumprod(data_np, axis=0)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumprod(data, axis=-1)
        z = np.cumprod(data_np, axis=-1)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumprod(data, dtype="float32")
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP32)

        y = paddle.cumprod(data, dtype=np.int32)
        self.assertTrue(y.dtype == core.VarDesc.VarType.INT32)

        y = paddle.cumprod(data, axis=-2)
        z = np.cumprod(data_np, axis=-2)
        self.assertTrue(np.array_equal(z, y.numpy()))

    def run_static(self, use_custom_device=False):
        with base.program_guard(base.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data("X", [100, 100])
            y = paddle.cumprod(x)
            y2 = paddle.cumprod(x, axis=0)
            y3 = paddle.cumprod(x, axis=-1)
            y4 = paddle.cumprod(x, dtype="float32")
            y5 = paddle.cumprod(x, dtype=np.int32)
            y6 = paddle.cumprod(x, axis=-2)

            place = base.CustomPlace("npu", 0) if use_custom_device else base.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            out = exe.run(
                feed={"X": data_np},
                fetch_list=[y.name, y2.name, y3.name, y4.name, y5.name, y6.name],
            )

            z = np.cumprod(data_np)
            self.assertTrue(np.allclose(z, out[0]))
            z = np.cumprod(data_np, axis=0)
            self.assertTrue(np.allclose(z, out[1]))
            z = np.cumprod(data_np, axis=-1)
            self.assertTrue(np.allclose(z, out[2]))
            self.assertTrue(out[3].dtype == np.float32)
            self.assertTrue(out[4].dtype == np.int32)
            z = np.cumprod(data_np, axis=-2)
            self.assertTrue(np.allclose(z, out[5]))

    def test_npu(self):
        # Now, npu tests need setting paddle.enable_static()

        self.run_static(use_custom_device=True)

    def test_name(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data("x", [3, 4])
            y = paddle.cumprod(x, name="out")
            self.assertTrue("out" in y.name)


class TestNPUCumprodOp1(OpTest):
    def setUp(self):
        self.op_type = "cumprod"
        self.set_npu()
        self.init_dtype()
        self.init_testcase()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_testcase(self):
        self.attrs = {"axis": 2}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumprod(axis=2)}


class TestNPUCumprodOp2(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": -1, "reverse": True}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {
            "Out": np.flip(np.flip(self.inputs["X"], axis=2).cumprod(axis=2), axis=2)
        }


class TestNPUCumprodOp3(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 1}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumprod(axis=1)}


class TestNPUCumprodOp4(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 0}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumprod(axis=0)}


class TestNPUCumprodOp5(TestNPUCumprodOp1):
    def init_testcase(self):
        self.inputs = {"X": np.random.random((5, 20)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumprod(axis=1)}


class TestNPUCumprodOp7(TestNPUCumprodOp1):
    def init_testcase(self):
        self.inputs = {"X": np.random.random((100)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumprod(axis=0)}


class TestNPUCumprodExclusive1(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((4, 5, 65)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumprod(axis=2)),
                axis=2,
            )
        }


class TestNPUCumprodExclusive2(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((1, 1, 888)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumprod(axis=2)),
                axis=2,
            )
        }


class TestNPUCumprodExclusive3(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((4, 5, 888)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumprod(axis=2)),
                axis=2,
            )
        }


class TestNPUCumprodExclusive4(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((1, 1, 3049)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumprod(axis=2)),
                axis=2,
            )
        }


class TestNPUCumprodExclusive5(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "exclusive": True}
        a = np.random.random((4, 5, 3096)).astype(self.dtype)
        self.inputs = {"X": a}
        self.outputs = {
            "Out": np.concatenate(
                (np.zeros((4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumprod(axis=2)),
                axis=2,
            )
        }


class TestNPUCumprodReverseExclusive(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": 2, "reverse": True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype(self.dtype)
        self.inputs = {"X": a}
        a = np.flip(a, axis=2)
        self.outputs = {
            "Out": np.concatenate(
                (
                    np.flip(a[:, :, :-1].cumprod(axis=2), axis=2),
                    np.zeros((4, 5, 1), dtype=self.dtype),
                ),
                axis=2,
            )
        }


class TestNPUCumprodWithFlatten1(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"flatten": True}
        self.inputs = {"X": np.random.random((5, 6)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumprod()}


class TestNPUCumprodWithFlatten2(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"flatten": True}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {"Out": self.inputs["X"].cumprod()}


# ----------------Cumprod Int64----------------
class TestNPUCumprodOpInt64(TestNPUCumprodOp1):
    def init_testcase(self):
        self.attrs = {"axis": -1, "reverse": True}
        self.inputs = {
            "X": np.random.randint(1, 10000, size=(5, 6, 10)).astype(self.dtype)
        }
        self.outputs = {
            "Out": np.flip(np.flip(self.inputs["X"], axis=2).cumprod(axis=2), axis=2)
        }


def create_test_int64(parent):
    class TestCumprodInt64(parent):
        def init_dtype(self):
            self.dtype = np.int64

    cls_name = "{0}_{1}".format(parent.__name__, "Int64")
    TestCumprodInt64.__name__ = cls_name
    globals()[cls_name] = TestCumprodInt64


create_test_int64(TestNPUCumprodOp1)
create_test_int64(TestNPUCumprodOp2)
create_test_int64(TestNPUCumprodOp3)
create_test_int64(TestNPUCumprodOp4)
create_test_int64(TestNPUCumprodOp5)
create_test_int64(TestNPUCumprodOp7)
create_test_int64(TestNPUCumprodExclusive1)
create_test_int64(TestNPUCumprodExclusive2)
create_test_int64(TestNPUCumprodExclusive3)
create_test_int64(TestNPUCumprodExclusive4)
create_test_int64(TestNPUCumprodExclusive5)
create_test_int64(TestNPUCumprodReverseExclusive)
create_test_int64(TestNPUCumprodWithFlatten1)
create_test_int64(TestNPUCumprodWithFlatten2)

if __name__ == "__main__":
    unittest.main()
