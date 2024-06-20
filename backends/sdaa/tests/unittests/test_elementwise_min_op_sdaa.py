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

paddle.enable_static()
SEED = 2021


@skip_check_grad_ci(reason="Haven not implement minimum grad kernel.")
class TestElementwiseMinOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_dtype()
        self.init_input_output()
        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {"Out": self.out}
        self.attrs = {"axis": self.axis}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_input_output(self):
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], [13, 17]).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(0.1, 1, [13, 17]).astype(
            self.dtype
        )
        self.out = np.minimum(self.x, self.y)
        self.axis = -1

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseMinOpFp16(TestElementwiseMinOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpDouble(TestElementwiseMinOp):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpInt32(TestElementwiseMinOp):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt64(TestElementwiseMinOp):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOp_ZeroDim1(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float32")
        self.y = np.random.uniform(0.1, 1, []).astype("float32")
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOp_ZeroDim2(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        self.y = np.random.uniform(0.1, 1, []).astype("float32")
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOp_ZeroDim3(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float32")
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOpFp16_ZeroDim1(TestElementwiseMinOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpFp16_ZeroDim2(TestElementwiseMinOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpFp16_ZeroDim3(TestElementwiseMinOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpDouble_ZeroDim1(TestElementwiseMinOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpDouble_ZeroDim2(TestElementwiseMinOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpDouble_ZeroDim3(TestElementwiseMinOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpInt32_ZeroDim1(TestElementwiseMinOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt32_ZeroDim2(TestElementwiseMinOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt32_ZeroDim3(TestElementwiseMinOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt64_ZeroDim1(TestElementwiseMinOp_ZeroDim1):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOpInt64_ZeroDim2(TestElementwiseMinOp_ZeroDim2):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOpInt64_ZeroDim3(TestElementwiseMinOp_ZeroDim3):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOp_scalar(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.random_integers(-5, 5, [10, 3, 4]).astype(self.dtype)
        self.y = np.array([0.5]).astype(self.dtype)
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOpFp16_scalar(TestElementwiseMinOp_scalar):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpDouble_scalar(TestElementwiseMinOp_scalar):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpInt32_scalar(TestElementwiseMinOp_scalar):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt64_scalar(TestElementwiseMinOp_scalar):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOp_Vector(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 2, (100,)).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(0.1, 1, (100,)).astype(
            self.dtype
        )
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOpFp16_Vector(TestElementwiseMinOp_Vector):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpDouble_Vector(TestElementwiseMinOp_Vector):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpInt32_Vector(TestElementwiseMinOp_Vector):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt64_Vector(TestElementwiseMinOp_Vector):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOp_broadcast(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], (100,)).astype(self.dtype)
        self.y = self.x[0, 0, :] + self.sgn * np.random.uniform(1, 2, (100,)).astype(
            self.dtype
        )
        self.out = np.minimum(self.x, self.y.reshape(1, 1, 100))
        self.axis = -1


class TestElementwiseMinOp_broadcast2(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 10, 2, 5)).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(
            self.dtype
        )
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOp_broadcast3(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 10, 1, 1)).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(
            self.dtype
        )
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOp_broadcast4(TestElementwiseMinOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 1, 2, 5)).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(
            self.dtype
        )
        self.out = np.minimum(self.x, self.y)
        self.axis = -1


class TestElementwiseMinOpFp16_broadcast(TestElementwiseMinOp_broadcast):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpFp16_broadcast2(TestElementwiseMinOp_broadcast2):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpFp16_broadcast3(TestElementwiseMinOp_broadcast3):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpFp16_broadcast4(TestElementwiseMinOp_broadcast4):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMinOpDouble_broadcast(TestElementwiseMinOp_broadcast):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpDouble_broadcast2(TestElementwiseMinOp_broadcast2):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpDouble_broadcast3(TestElementwiseMinOp_broadcast3):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpDouble_broadcast4(TestElementwiseMinOp_broadcast4):
    def init_dtype(self):
        self.dtype = np.double


class TestElementwiseMinOpInt32_broadcast(TestElementwiseMinOp_broadcast):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt32_broadcast2(TestElementwiseMinOp_broadcast2):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt32_broadcast3(TestElementwiseMinOp_broadcast3):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt32_broadcast4(TestElementwiseMinOp_broadcast4):
    def init_dtype(self):
        self.dtype = np.int32


class TestElementwiseMinOpInt64_broadcast(TestElementwiseMinOp_broadcast):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOpInt64_broadcast2(TestElementwiseMinOp_broadcast2):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOpInt64_broadcast3(TestElementwiseMinOp_broadcast3):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseMinOpInt64_broadcast4(TestElementwiseMinOp_broadcast4):
    def init_dtype(self):
        self.dtype = np.int64


# This case comes from UIE-X-base model
@skip_check_grad_ci(reason="Haven not implement minimum grad kernel.")
class TestElementwiseMinOpINT64(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_dtype()
        self.init_input_output()
        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {"Out": self.out}
        self.attrs = {"axis": self.axis}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_input_output(self):
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [4, 512, 512]).astype(self.dtype)
        self.sgn = np.random.choice([-1, 1], [4, 512, 512]).astype(self.dtype)
        self.y = self.x + self.sgn * np.random.uniform(0.1, 1, [4, 512, 512]).astype(
            self.dtype
        )
        self.out = np.minimum(self.x, self.y)
        self.axis = -1

    def init_dtype(self):
        self.dtype = np.int64

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseMinOpNet(unittest.TestCase):
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

            c = paddle.minimum(a, b)

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
