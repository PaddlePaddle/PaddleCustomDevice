# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import unittest
from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2022


def create_test_fp16_class(parent):
    class TestFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16
            self.shape = [3, 3]
            self.axis = -1

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


class TestSoftmax(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "softmax"
        self.python_api = paddle.nn.functional.softmax
        self.init_dtype()

        x = np.random.random(self.shape).astype(self.dtype)
        np_out = np.exp(x) / np.sum(np.exp(x), axis=self.axis, keepdims=True)
        self.inputs = {"X": x}
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": np_out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float32
        self.shape = [3, 3]
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
            atol=1e-3 if self.dtype == np.float16 else 1e-5,
        )

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                paddle.CustomPlace("sdaa", 0),
                ["X"],
                "Out",
                max_relative_error=0.01,
            )
        else:
            self.check_grad_with_place(
                paddle.CustomPlace("sdaa", 0),
                ["X"],
                "Out",
                numeric_place=paddle.CPUPlace(),
            )


class TestSoftmaxAxis(TestSoftmax):
    def init_dtype(self):
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.axis = 0


class TestSoftmaxAxis1(TestSoftmax):
    def init_dtype(self):
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.axis = 1


class TestSoftmaxAxis1(TestSoftmax):
    def init_dtype(self):
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.axis = 2


create_test_fp16_class(TestSoftmax)


class TestSoftmaxNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)
        a_np = np.random.random(size=(4, 32)).astype("float32")
        b_np = np.random.random(size=(4, 32)).astype("float32")
        label_np = np.random.randint(2, size=(4, 1)).astype("int64")
        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[4, 32], dtype="float32")
            b = paddle.static.data(name="b", shape=[4, 32], dtype="float32")
            label = paddle.static.data(name="label", shape=[4, 1], dtype="int64")
            c = paddle.multiply(a, b)
            d = paddle.sqrt(c)
            # 4 x 128
            fc_1 = paddle.static.nn.fc(x=d, size=128)
            # 4 x 2
            prediction = paddle.static.nn.fc(x=fc_1, size=2)

            # 4 x 2
            prob = paddle.nn.functional.softmax(prediction, axis=1)

            cost = paddle.nn.functional.cross_entropy(input=prob, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.Momentum(learning_rate=0.01)
            sgd.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)
        for epoch in range(100):
            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "label": label_np},
                fetch_list=[prediction, loss],
            )
            if epoch % 10 == 0:
                print(
                    "Epoch {} | Prediction[0]: {}, Loss: {}".format(
                        epoch, pred_res[0], loss_res
                    )
                )

        return pred_res, loss_res

    def test_sdaa(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred, rtol=1e-2))
        self.assertTrue(np.allclose(npu_loss, cpu_loss, rtol=1e-2))


class TestSoftmaxHighPrecision(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "softmax"
        self.python_api = paddle.nn.functional.softmax
        self.init_dtype()

        x = np.random.random(self.shape).astype(self.dtype)
        np_out = np.exp(x) / np.sum(np.exp(x), axis=self.axis, keepdims=True)
        self.inputs = {"X": x}
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": np_out}

    def set_sdaa(self):
        os.environ["HIGH_PRECISION_OP_LIST"] = "softmax,softmax_grad"
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32
        self.shape = [10, 10]
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                paddle.CustomPlace("sdaa", 0),
                ["X"],
                "Out",
            )
        else:
            self.check_grad_with_place(
                paddle.CustomPlace("sdaa", 0),
                ["X"],
                "Out",
                numeric_place=paddle.CPUPlace(),
            )


if __name__ == "__main__":
    unittest.main()
