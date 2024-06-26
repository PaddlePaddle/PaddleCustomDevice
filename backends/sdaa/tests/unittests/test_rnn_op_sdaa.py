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

import random
import sys
import unittest

import numpy as np
from op_test import OpTest

import paddle

sys.path.append("../rnn")
from convert import get_params_for_net
from rnn_numpy import LSTM

random.seed(2023)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()

SEED = 2024


def rnn_wrapper(
    Input,
    PreState,
    WeightList=None,
    SequenceLength=None,
    dropout_prob=0.0,
    is_bidirec=False,
    input_size=10,
    hidden_size=100,
    num_layers=1,
    mode="LSTM",
    seed=0,
    is_test=False,
):
    dropout_state_in = paddle.Tensor()
    return paddle._C_ops.rnn(
        Input,
        PreState,
        WeightList,
        SequenceLength,
        dropout_state_in,
        dropout_prob,
        is_bidirec,
        input_size,
        hidden_size,
        num_layers,
        mode,
        seed,
        is_test,
    )


class TestRNNOp(OpTest):
    def get_weight_names(self):
        weight_names = []
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append(f"{i}.weight_{j}")
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append(f"{i}.bias_{j}")
        return weight_names

    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.op_type = "rnn"
        self.python_api = rnn_wrapper
        self.python_out_sig = ["Out", "DropoutState", "State"]
        self.python_out_sig_sub_name = {"State": ["last_hidden", "last_cell"]}
        self.dtype = np.float32
        self.sequence_length = None
        self.num_layers = 1
        self.is_bidirec = False
        self.mode = "LSTM"
        self.is_test = False
        self.dropout = 0.0
        self.set_attrs()

        self.direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"
        seq_length = 12
        batch_size = 5
        input_size = 3
        hidden_size = 2

        input = np.random.uniform(
            low=-0.1, high=0.1, size=(seq_length, batch_size, input_size)
        ).astype(self.dtype)
        if self.sequence_length is not None:
            input[11][1:][:] = 0
            input[10][2:][:] = 0
            input[9][3:][:] = 0
            input[8][4:][:] = 0

        rnn1 = LSTM(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction=direction,
            dropout=self.dropout,
            dtype=self.dtype,
        )

        flat_w = get_params_for_net(rnn1)
        output, (last_hidden, last_cell) = rnn1(
            input, sequence_length=self.sequence_length
        )

        init_h = np.zeros(
            (self.num_layers * self.direction_num, batch_size, hidden_size)
        ).astype(self.dtype)
        init_c = np.zeros(
            (self.num_layers * self.direction_num, batch_size, hidden_size)
        ).astype(self.dtype)
        state_out = np.ndarray(300).astype("uint8")

        self.inputs = {
            "Input": input,
            "WeightList": flat_w,
            "PreState": [("init_h", init_h), ("init_c", init_c)],
            "SequenceLength": self.sequence_length,
        }
        if self.sequence_length is None:
            self.inputs = {
                "Input": input,
                "WeightList": flat_w,
                "PreState": [("init_h", init_h), ("init_c", init_c)],
            }
        self.attrs = {
            "dropout_prob": self.dropout,
            "is_bidirec": self.is_bidirec,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": self.num_layers,
            "mode": self.mode,
            "is_test": self.is_test,
        }
        self.outputs = {
            "Out": output,
            "State": [("last_hidden", last_hidden), ("last_cell", last_cell)],
            "Reserve": np.ndarray(400).astype("uint8"),
            "DropoutState": state_out,
        }

    def test_output(self):
        self.check_output_with_place(
            self.place, no_check_set=["Reserve", "DropoutState"], atol=1e-7
        )

    def set_attrs(self):
        pass

    def test_grad(self):
        if not self.is_test:
            var_name_list = self.get_weight_names()
            grad_check_list = ["Input", "init_h", "init_c"]
            grad_check_list.extend(var_name_list)
            self.check_grad_with_place(
                self.place,
                set(grad_check_list),
                ["Out", "last_hidden", "last_cell"],
                numeric_place=paddle.CPUPlace(),
            )

    def test_grad_only_input(self):
        if not self.is_test:
            var_name_list = self.get_weight_names()
            grad_check_list = ["Input"]
            grad_check_list.extend(var_name_list)
            self.check_grad_with_place(
                self.place,
                set(grad_check_list),
                ["Out", "last_hidden", "last_cell"],
                numeric_place=paddle.CPUPlace(),
            )

    def test_grad_only_h(self):
        if not self.is_test:
            var_name_list = self.get_weight_names()
            grad_check_list = ["init_h"]
            grad_check_list.extend(var_name_list)
            self.check_grad_with_place(
                self.place,
                set(grad_check_list),
                ["Out", "last_hidden", "last_cell"],
                numeric_place=paddle.CPUPlace(),
            )

    def test_grad_only_c(self):
        if not self.is_test:
            var_name_list = self.get_weight_names()
            grad_check_list = ["init_c"]
            grad_check_list.extend(var_name_list)
            self.check_grad_with_place(
                self.place,
                set(grad_check_list),
                ["Out", "last_hidden", "last_cell"],
                numeric_place=paddle.CPUPlace(),
            )


@unittest.skip("paddle-sdaa do not support variable length inputs")
class TestRNNOp1(TestRNNOp):
    def set_attrs(self):
        self.sequence_length = np.array([12, 11, 10, 9, 8], dtype=np.int32)


class TestRNNOp2(TestRNNOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_bidirec = True


@unittest.skip("tecodnn not support inference for RNN")
class TestRNNOp3(TestRNNOp):
    def set_attrs(self):
        self.is_test = True
        self.sequence_length = None


@unittest.skip("tecodnn not support inference for RNN")
class TestRNNOp4(TestRNNOp):
    def set_attrs(self):
        self.is_test = True
        self.sequence_length = None
        self.is_bidirec = True


class TestRNNOp5(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 2


class TestRNNOp6(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 2
        self.is_bidirec = True


@unittest.skip("tecodnn not support inference for RNN")
class TestRNNOp7(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 2
        self.is_bidirec = True
        self.is_test = True


class TestRNNOp8(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 2
        self.is_bidirec = True
        self.sequence_length = None


class TestRNNOp9(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 3


class TestRNNOpError(unittest.TestCase):
    def test_errors(self):
        def test_variable_sequence_len():
            paddle.device.set_device("sdaa")
            x = paddle.randn((4, 10, 16))
            x.stop_gradient = False
            prev_h = paddle.randn((4, 4, 32))
            prev_c = paddle.randn((4, 4, 32))
            seq_len = paddle.to_tensor(np.array([10, 6, 8, 5]))
            rnn = paddle.nn.LSTM(16, 32, 2, direction="bidirectional")
            y, (h, c) = rnn(x, (prev_h, prev_c), seq_len)

        paddle.disable_static()
        self.assertRaises(ValueError, test_variable_sequence_len)
        paddle.enable_static()


class TestLSTMNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        x_np = np.random.random(size=(4, 10, 16)).astype("float32")
        prev_h_np = np.random.random(size=(4, 4, 32)).astype("float32")
        prev_c_np = np.random.random(size=(4, 4, 32)).astype("float32")

        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data(name="x", shape=[4, 10, 16], dtype="float32")
            prev_h = paddle.static.data(
                name="prev_h", shape=[4, 4, 32], dtype="float32"
            )
            prev_c = paddle.static.data(
                name="prev_c", shape=[4, 4, 32], dtype="float32"
            )

            rnn = paddle.nn.LSTM(16, 32, 2, direction="bidirectional")
            y, (h, c) = rnn(x, (prev_h, prev_c))
            y = paddle.static.nn.fc(x=y, size=128)
            y = paddle.static.nn.fc(x=y, size=2, activation="softmax")

            loss = paddle.mean(y)

            optimizer = paddle.optimizer.Momentum(learning_rate=0.1)
            optimizer.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))

        pred_res, loss_res = exe.run(
            main_prog,
            feed={"x": x_np, "prev_h": prev_h_np, "prev_c": prev_c_np},
            fetch_list=[y, loss],
        )

        print("Prediction[0]: {}, Loss: {}".format(pred_res[0], loss_res))

        return pred_res, loss_res

    def test_sdaa(self):
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss))


if __name__ == "__main__":
    unittest.main()
