#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import math
import paddle
import random
from tests.op_test import OpTest

random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()


def get_params_for_cell(np_cell, num_layers, idx):
    state = np_cell.parameters
    weight_list = [
        ("{}.weight_{}".format(num_layers, idx), state["weight_ih"]),
        ("{}.weight_{}".format(num_layers, idx + 1), state["weight_hh"]),
    ]
    bias_list = [
        ("{}.bias_{}".format(num_layers, idx), state["bias_ih"]),
        ("{}.bias_{}".format(num_layers, idx + 1), state["bias_hh"]),
    ]
    return weight_list, bias_list


def get_params_for_net(np_net):
    weight_list = []
    bias_list = []
    for layer_idx, np_layer in enumerate(np_net):
        if hasattr(np_layer, "cell"):
            weight, bias = get_params_for_cell(np_layer.cell, layer_idx, 0)
            for w, b in zip(weight, bias):
                weight_list.append(w)
                bias_list.append(b)
        else:
            for count, cell in enumerate([np_layer.cell_fw, np_layer.cell_bw]):
                weight, bias = get_params_for_cell(cell, layer_idx, count * 2)
                for w, b in zip(weight, bias):
                    weight_list.append(w)
                    bias_list.append(b)

    weight_list.extend(bias_list)
    return weight_list


def unstack(array, axis=0):
    num = array.shape[axis]
    sub_arrays = np.split(array, num, axis)
    return [np.squeeze(sub_array, axis) for sub_array in sub_arrays]


def dropout(array, p=0.5):
    if p == 0.0:
        return array
    mask = (np.random.uniform(size=array.shape) < (1 - p)).astype(array.dtype)
    return array * (mask / (1 - p))


def flatten(nested):
    return list(_flatten(nested))


def _flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            for subitem in _flatten(item):
                yield subitem
        else:
            yield item


def concat_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        return np.stack(flatten(states))
    else:
        states = flatten(states)
        componnets = []
        for i in range(state_components):
            componnets.append(states[i::state_components])
        return [np.stack(item) for item in componnets]


def split_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        states = unstack(states)
        if not bidirectional:
            return states
        else:
            return list(zip(states[::2], states[1::2]))
    else:
        assert len(states) == state_components
        states = tuple([unstack(item) for item in states])
        if not bidirectional:
            return list(zip(*states))
        else:
            states = list(zip(*states))
            return list(zip(states[::2], states[1::2]))


def sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = np.max(lengths)
    else:
        assert max_len >= np.max(lengths)
    return np.arange(max_len) < np.expand_dims(lengths, -1)


def update_state(mask, new, old):
    if not isinstance(old, (tuple, list)):
        return np.where(mask, new, old)
    else:
        return tuple(map(lambda x, y: np.where(mask, x, y), new, old))


def rnn(
    cell,
    inputs,
    initial_states,
    sequence_length=None,
    time_major=False,
    is_reverse=False,
):
    if not time_major:
        inputs = np.transpose(inputs, [1, 0, 2])
    if is_reverse:
        inputs = np.flip(inputs, 0)

    if initial_states is None:
        initial_states = cell.init_state(inputs, 1)

    if sequence_length is None:
        mask = None
    else:
        mask = np.transpose(sequence_mask(sequence_length), [1, 0])
        mask = np.expand_dims(mask, -1)
        if is_reverse:
            mask = np.flip(mask, 0)

    time_steps = inputs.shape[0]
    state = initial_states
    outputs = []
    for t in range(time_steps):
        x_t = inputs[t]
        if mask is not None:
            m_t = mask[t]
            y, new_state = cell(x_t, state)
            y = np.where(m_t, y, 0.0)
            outputs.append(y)
            state = update_state(m_t, new_state, state)
        else:
            y, new_state = cell(x_t, state)
            outputs.append(y)
            state = new_state

    outputs = np.stack(outputs)
    final_state = state

    if is_reverse:
        outputs = np.flip(outputs, 0)
    if not time_major:
        outputs = np.transpose(outputs, [1, 0, 2])
    return outputs, final_state


def birnn(
    cell_fw, cell_bw, inputs, initial_states, sequence_length=None, time_major=False
):
    states_fw, states_bw = initial_states
    outputs_fw, states_fw = rnn(
        cell_fw, inputs, states_fw, sequence_length, time_major=time_major
    )

    outputs_bw, states_bw = rnn(
        cell_bw,
        inputs,
        states_bw,
        sequence_length,
        time_major=time_major,
        is_reverse=True,
    )

    outputs = np.concatenate((outputs_fw, outputs_bw), -1)
    final_states = (states_fw, states_bw)
    return outputs, final_states


class LayerMixin(object):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class LayerListMixin(LayerMixin):
    def __init__(self, layers=None):
        self._layers = list(layers) if layers else []

    def append(self, layer):
        self._layers.append(layer)

    def __iter__(self):
        return iter(self._layers)


class LSTMCell(LayerMixin):
    def __init__(self, input_size, hidden_size, bias=True, dtype="float64"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.parameters = dict()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = np.random.uniform(
            -std, std, (4 * hidden_size, input_size)
        ).astype(dtype)
        self.weight_hh = np.random.uniform(
            -std, std, (4 * hidden_size, hidden_size)
        ).astype(dtype)
        self.parameters["weight_ih"] = self.weight_ih
        self.parameters["weight_hh"] = self.weight_hh
        if bias:
            self.bias_ih = np.random.uniform(-std, std, (4 * hidden_size)).astype(dtype)
            self.bias_hh = np.random.uniform(-std, std, (4 * hidden_size)).astype(dtype)
            self.parameters["bias_ih"] = self.bias_ih
            self.parameters["bias_hh"] = self.bias_hh
        else:
            self.bias_ih = None
            self.bias_hh = None

    def init_state(self, inputs, batch_dim_index=0):
        batch_size = inputs.shape[batch_dim_index]
        init_h = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        init_c = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        return init_h, init_c

    def forward(self, inputs, hx=None):
        if hx is None:
            hx = self.init_state(inputs)
        pre_hidden, pre_cell = hx
        gates = np.matmul(inputs, self.weight_ih.T)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += np.matmul(pre_hidden, self.weight_hh.T)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        chunked_gates = np.split(gates, 4, -1)

        i = 1.0 / (1.0 + np.exp(-chunked_gates[0]))
        f = 1.0 / (1.0 + np.exp(-chunked_gates[1]))
        o = 1.0 / (1.0 + np.exp(-chunked_gates[3]))
        c = f * pre_cell + i * np.tanh(chunked_gates[2])
        h = o * np.tanh(c)

        return h, (h, c)


class RNNMixin(LayerListMixin):
    def forward(self, inputs, initial_states=None, sequence_length=None):
        batch_index = 1 if self.time_major else 0
        batch_size = inputs.shape[batch_index]
        dtype = inputs.dtype
        if initial_states is None:
            state_shape = (
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
            )
            if self.state_components == 1:
                initial_states = np.zeros(state_shape, dtype)
            else:
                initial_states = tuple(
                    [np.zeros(state_shape, dtype) for _ in range(self.state_components)]
                )

        states = split_states(
            initial_states, self.num_directions == 2, self.state_components
        )
        final_states = []
        input_temp = inputs
        for i, rnn_layer in enumerate(self):
            if i > 0:
                input_temp = dropout(inputs, self.dropout)
            outputs, final_state = rnn_layer(input_temp, states[i], sequence_length)
            final_states.append(final_state)
            inputs = outputs

        final_states = concat_states(
            final_states, self.num_directions == 2, self.state_components
        )
        return outputs, final_states


class RNN(LayerMixin):
    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            # for non-dygraph mode, `rnn` api uses cell.call
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major

    def forward(self, inputs, initial_states=None, sequence_length=None):
        final_outputs, final_states = rnn(
            self.cell,
            inputs,
            initial_states=initial_states,
            sequence_length=sequence_length,
            time_major=self.time_major,
            is_reverse=self.is_reverse,
        )
        return final_outputs, final_states


class BiRNN(LayerMixin):
    def __init__(self, cell_fw, cell_bw, time_major=False):
        super(BiRNN, self).__init__()
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw
        self.time_major = time_major

    def forward(self, inputs, initial_states=None, sequence_length=None, **kwargs):
        if isinstance(initial_states, (list, tuple)):
            assert (
                len(initial_states) == 2
            ), "length of initial_states should be 2 when it is a list/tuple"
        else:
            initial_states = [initial_states, initial_states]

        outputs, final_states = birnn(
            self.cell_fw,
            self.cell_bw,
            inputs,
            initial_states,
            sequence_length,
            self.time_major,
        )
        return outputs, final_states


class LSTM(RNNMixin):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        direction="forward",
        dropout=0.0,
        time_major=False,
        dtype="float64",
    ):
        super(LSTM, self).__init__()

        bidirectional_list = ["bidirectional", "bidirect"]
        if direction in ["forward"]:
            is_reverse = False
            cell = LSTMCell(input_size, hidden_size, dtype=dtype)
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = LSTMCell(hidden_size, hidden_size, dtype=dtype)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction in bidirectional_list:
            cell_fw = LSTMCell(input_size, hidden_size, dtype=dtype)
            cell_bw = LSTMCell(input_size, hidden_size, dtype=dtype)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = LSTMCell(2 * hidden_size, hidden_size, dtype=dtype)
                cell_bw = LSTMCell(2 * hidden_size, hidden_size, dtype=dtype)
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward, backward or bidirectional, "
                "received direction = {}".format(direction)
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 2


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
        self.place = paddle.CustomPlace("mlu", 0)
        self.__class__.use_custom_device = True
        self.in_type = np.float32
        self.init_dtype()
        self.init_size()
        self.op_type = "rnn"
        self.sequence_length = np.array([12, 11, 10, 9, 8], dtype=np.int32)
        self.num_layers = 1
        self.is_bidirec = False
        self.mode = "LSTM"
        self.is_test = False
        self.dropout = 0.0
        self.set_attrs()

        self.direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"

        input = np.random.uniform(
            low=-0.1, high=0.1, size=(self.seq_length, self.batch_size, self.input_size)
        ).astype(self.dtype)

        input[11][1:][:] = 0
        input[10][2:][:] = 0
        input[9][3:][:] = 0
        input[8][4:][:] = 0

        rnn1 = LSTM(
            self.input_size,
            self.hidden_size,
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
            (self.num_layers * self.direction_num, self.batch_size, self.hidden_size)
        ).astype(self.dtype)
        init_c = np.zeros(
            (self.num_layers * self.direction_num, self.batch_size, self.hidden_size)
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
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
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

    def init_dtype(self):
        self.dtype = self.in_type

    def init_size(self):
        self.seq_length = 12
        self.batch_size = 5
        self.input_size = 3
        self.hidden_size = 2

    def test_output(self):
        self.check_output_with_place(
            self.place, atol=1e-4, no_check_set=["Reserve", "DropoutState", "State"]
        )

    def set_attrs(self):
        pass

    def test_grad(self):
        if not self.is_test and self.sequence_length is None:
            # if not self.is_test:
            var_name_list = self.get_weight_names()
            grad_check_list = ["Input", "init_h", "init_c"]
            grad_check_list.extend(var_name_list)
            self.check_grad_with_place(
                self.place, set(grad_check_list), ["Out", "last_hidden", "last_cell"]
            )


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


# TODO(chenxiao): cnnl doesn't support num_layers > 1 case
# class TestRNNOp5(TestRNNOp):

#     def set_attrs(self):
#         self.num_layers = 2

# class TestRNNOp6(TestRNNOp):

#     def set_attrs(self):
#         self.num_layers = 2
#         self.is_bidirec = True

# class TestRNNOp7(TestRNNOp):

#     def set_attrs(self):
#         self.num_layers = 2
#         self.is_bidirec = True
#         self.is_test = True

# class TestRNNOp8(TestRNNOp):

#     def set_attrs(self):
#         self.num_layers = 2
#         self.is_bidirec = True
#         self.sequence_length = None

# class TestRNNOp9(TestRNNOp):

#     def set_attrs(self):
#         self.num_layers = 3

if __name__ == "__main__":
    unittest.main()
