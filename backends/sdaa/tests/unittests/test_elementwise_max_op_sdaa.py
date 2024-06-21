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

import numpy as np
import unittest
from op_test import OpTest
import paddle
from op_test import skip_check_grad_ci

paddle.enable_static()
SEED = 1234


@skip_check_grad_ci(reason="Haven not implement maximum grad kernel.")
class TestElementwiseMaxOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elementwise_max"
        self.python_api = paddle.maximum
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {"Out": self.out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseMaxOp_Fp16(TestElementwiseMaxOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMaxOp_Double(TestElementwiseMaxOp):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMaxOp_Int32(TestElementwiseMaxOp):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMaxOp_Int64(TestElementwiseMaxOp):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMaxOp_ZeroDim1(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float32")
        self.y = np.random.uniform(0.1, 1, []).astype("float32")
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_ZeroDim2(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        self.y = np.random.uniform(0.1, 1, []).astype("float32")
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_ZeroDim3(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float32")
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOpFp16_ZeroDim1(TestElementwiseMaxOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMaxOpFp16_ZeroDim2(TestElementwiseMaxOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMaxOpFp16_ZeroDim3(TestElementwiseMaxOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMaxOpDouble_ZeroDim1(TestElementwiseMaxOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMaxOpDouble_ZeroDim2(TestElementwiseMaxOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMaxOpDouble_ZeroDim3(TestElementwiseMaxOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMaxOpInt32_ZeroDim1(TestElementwiseMaxOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMaxOpInt32_ZeroDim2(TestElementwiseMaxOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMaxOpInt32_ZeroDim3(TestElementwiseMaxOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMaxOpInt64_ZeroDim1(TestElementwiseMaxOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMaxOpInt64_ZeroDim2(TestElementwiseMaxOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMaxOpInt64_ZeroDim3(TestElementwiseMaxOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMaxOp_scalar(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.random_integers(-5, 5, [2, 3, 20]).astype(self.dtype)
        self.y = np.array([0.5]).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOpFp16_scalar(TestElementwiseMaxOp_scalar):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMaxOpDouble_scalar(TestElementwiseMaxOp_scalar):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMaxOpInt32_scalar(TestElementwiseMaxOp_scalar):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMaxOpInt64_scalar(TestElementwiseMaxOp_scalar):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMaxOp_Vector(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, (100,)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOpFp16_Vector(TestElementwiseMaxOp_Vector):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMaxOpDouble_Vector(TestElementwiseMaxOp_Vector):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMaxOpInt32_Vector(TestElementwiseMaxOp_Vector):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMaxOpInt64_Vector(TestElementwiseMaxOp_Vector):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMaxOp_broadcast_0(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (100, 5, 2)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
        self.y = (
            self.x[:, 0, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(self.dtype)
        ).reshape(100, 1, 1)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_1(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 100, 3)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
        self.y = (
            self.x[0, :, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(self.dtype)
        ).reshape(1, 100, 1)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_2(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
        self.y = (
            self.x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(self.dtype)
        ).reshape(1, 1, 100)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_3(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 50, 2, 1)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (50, 2)).astype(self.dtype)
        self.y = (
            self.x[0, :, :, 0]
            + sgn * np.random.uniform(1, 2, (50, 2)).astype(self.dtype)
        ).reshape(1, 50, 2, 1)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_4(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 5)).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(1, 2, (2, 3, 1, 5)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_5(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 1)).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(1, 2, (2, 3, 1, 1)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_6(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (1, 3, 1, 1)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 1)).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(1, 2, (2, 3, 1, 1)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_7(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (1, 3, 4, 1)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 1)).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(1, 2, (2, 3, 1, 1)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 32)).astype("float32")
        b_np = np.random.random(size=(32, 32)).astype("float32")
        label_np = np.random.randint(2, size=(32, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype="float32")
            b = paddle.static.data(name="b", shape=[32, 32], dtype="float32")
            label = paddle.static.data(name="label", shape=[32, 1], dtype="int64")

            c = paddle.maximum(a, b)

            fc_1 = paddle.static.nn.fc(x=c, size=128)
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation="softmax")

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.Momentum(learning_rate=0.01)
            sgd.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
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
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss))


if __name__ == "__main__":
    unittest.main()
