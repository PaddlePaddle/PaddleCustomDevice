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

import random
import unittest
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

import numpy as np
from tests.op_test import OpTest

import paddle

from tests.convert import get_params_for_net
from tests.rnn_numpy import LSTM, GRU

random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()


def rnn_warpper(
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
                weight_names.append("{}.weight_{}".format(i, j))
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append("{}.bias_{}".format(i, j))
        return weight_names

    def setUp(self):
        self.set_npu()
        self.op_type = "rnn"
        self.python_api = rnn_warpper
        self.python_out_sig = ["Out", "DropoutState", "State"]
        self.python_out_sig_sub_name = {"State": ["last_hidden", "last_cell"]}
        self.dtype = np.float32
        self.sequence_length = np.array([12, 11, 10, 9, 8], dtype=np.int32)
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
        state_out = np.ndarray((300)).astype("uint8")

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
            "Reserve": np.ndarray((400)).astype("uint8"),
            "DropoutState": state_out,
        }

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", select_npu)

    def test_check_output(self):
        self.check_output_with_place(
            self.place, no_check_set=["Reserve", "DropoutState"], atol=1e-3
        )

    def set_attrs(self):
        pass

    # TODO(songkai05):implement gradient check
    # def test_check_grad(self):
    #     if not self.is_test:
    #         var_name_list = self.get_weight_names()
    #         grad_check_list = ['Input', 'init_h', 'init_c']
    #         grad_check_list.extend(var_name_list)
    #         self.check_grad_with_place(
    #             self.place, set(grad_check_list), ['Out', 'last_hidden', 'last_cell']
    #         )

    # def test_grad_only_input(self):
    #     if not self.is_test:
    #         var_name_list = self.get_weight_names()
    #         grad_check_list = ['Input']
    #         grad_check_list.extend(var_name_list)
    #         self.check_grad_with_place(
    #             self.place, set(grad_check_list), ['Out', 'last_hidden', 'last_cell']
    #         )

    # def test_grad_only_h(self):
    #     if not self.is_test:
    #         var_name_list = self.get_weight_names()
    #         grad_check_list = ['init_h']
    #         grad_check_list.extend(var_name_list)
    #         self.check_grad_with_place(
    #             self.place, set(grad_check_list), ['Out', 'last_hidden', 'last_cell']
    #         )

    # def test_grad_only_c(self):
    #     if not self.is_test:
    #         var_name_list = self.get_weight_names()
    #         grad_check_list = ['init_c']
    #         grad_check_list.extend(var_name_list)
    #         self.check_grad_with_place(
    #             self.place, set(grad_check_list), ['Out', 'last_hidden', 'last_cell']
    #         )


class TestRNNOp1(TestRNNOp):
    def set_attrs(self):
        self.sequence_length = None


class TestRNNOp2(TestRNNOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_bidirec = True


class TestRNNOp3(TestRNNOp):
    def set_attrs(self):
        self.is_test = True
        self.sequence_length = None


class TestRNNOp4(TestRNNOp):
    def set_attrs(self):
        self.is_test = True
        self.sequence_length = None
        self.is_bidirec = True


class TestRNNOp5(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 2


# TODO(songkai05):improve multi-layers + bidirec precision
# class TestRNNOp6(TestRNNOp):
#     def set_attrs(self):
#         self.num_layers = 2
#         self.is_bidirec = True


# class TestRNNOp7(TestRNNOp):
#     def set_attrs(self):
#         self.num_layers = 2
#         self.is_bidirec = True
#         self.is_test = True


class TestRNNOp8(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 2
        self.is_bidirec = True
        self.sequence_length = None


class TestRNNOp9(TestRNNOp):
    def set_attrs(self):
        self.num_layers = 3


class TestGRUOp(OpTest):
    def get_weight_names(self):
        weight_names = []
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append("{}.weight_{}".format(i, j))
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append("{}.bias_{}".format(i, j))
        return weight_names

    def setUp(self):
        self.set_npu()
        self.op_type = "rnn"
        self.dtype = "float32"
        self.sequence_length = (
            None
            # np.array([12, 11, 10, 9, 8, 7, 6, 5], dtype=np.int32)
        )
        self.num_layers = 1
        self.is_bidirec = False
        self.is_test = False
        self.mode = "GRU"
        self.dropout = 0.0
        seq_length = 12
        batch_size = 8
        input_size = 4
        self.hidden_size = 2
        self.set_attrs()

        self.direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"

        input = np.random.uniform(
            low=-0.1, high=0.1, size=(seq_length, batch_size, input_size)
        ).astype(self.dtype)

        if self.sequence_length is not None:
            input[3][1:][:] = 0
            input[4][2:][:] = 0
            input[2][3:][:] = 0
            input[1][4:][:] = 0

        rnn1 = GRU(
            input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction=direction,
            dropout=self.dropout,
            dtype=self.dtype,
        )

        flat_w = get_params_for_net(rnn1)

        output, last_hidden = rnn1(input, sequence_length=self.sequence_length)

        init_h = np.zeros(
            (self.num_layers * self.direction_num, batch_size, self.hidden_size)
        ).astype(self.dtype)

        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            "Input": input,
            "WeightList": flat_w,
            "PreState": [("init_h", init_h)],
            "SequenceLength": self.sequence_length,
        }
        if self.sequence_length is None:
            self.inputs = {
                "Input": input,
                "WeightList": flat_w,
                "PreState": [("init_h", init_h)],
            }
        self.attrs = {
            "dropout_prob": self.dropout,
            "is_bidirec": self.is_bidirec,
            "input_size": input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "is_test": self.is_test,
            "mode": self.mode,
        }
        self.outputs = {
            "Out": output,
            "State": [("last_hidden", last_hidden)],
            "Reserve": np.ndarray((400)).astype("uint8"),
            "DropoutState": state_out,
        }

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", select_npu)

    def set_attrs(self):
        pass

    def test_output(self):
        self.check_output_with_place(
            self.place, no_check_set=["Reserve", "DropoutState"], atol=1e-3
        )

    # def test_grad(self):
    #     if not self.is_test:
    #         var_name_list = self.get_weight_names()
    #         grad_check_list = ['Input', 'init_h']
    #         grad_check_list.extend(var_name_list)
    #         self.check_grad_with_place(self.place, set(grad_check_list), ['Out', 'last_hidden'])


class TestGRUOp1(TestGRUOp):
    def set_attrs(self):
        self.sequence_length = None


class TestGRUOp2(TestGRUOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_bidirec = True


class TestGRUOp3(TestGRUOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_test = True


class TestGRUOp4(TestGRUOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_bidirec = True
        self.is_test = True


class TestGRUOpAvx(TestGRUOp):
    def set_attrs(self):
        self.dtype = "float32"
        self.hidden_size = 8


if __name__ == "__main__":
    unittest.main()
