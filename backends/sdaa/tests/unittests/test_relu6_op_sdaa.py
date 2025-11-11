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

import numpy as np
import unittest

from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2021


class TestRelu6(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "relu6"
        self.python_api = paddle.nn.functional.relu6
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)

        x = np.random.uniform(-10, 10, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.minimum(np.maximum(x, 0), 6)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestRelu6Fp16(TestRelu6):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.006,
        )


class TestRelu6Net(unittest.TestCase):
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

            sum = paddle.add(a, b)
            z = paddle.nn.functional.relu6(sum)

            fc_1 = paddle.static.nn.fc(x=z, size=128)
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
