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

import paddle


def convert_params_for_cell(np_cell, paddle_cell):
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        v.set_value(state[k])


def convert_params_for_cell_static(np_cell, paddle_cell, place):
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        scope = paddle.static.global_scope()
        tensor = scope.find_var(v.name).get_tensor()
        tensor.set(state[k], place)


def convert_params_for_net(np_net, paddle_net):
    for np_layer, paddle_layer in zip(np_net, paddle_net):
        if hasattr(np_layer, "cell"):
            convert_params_for_cell(np_layer.cell, paddle_layer.cell)
        else:
            convert_params_for_cell(np_layer.cell_fw, paddle_layer.cell_fw)
            convert_params_for_cell(np_layer.cell_bw, paddle_layer.cell_bw)


def convert_params_for_net_static(np_net, paddle_net, place):
    for np_layer, paddle_layer in zip(np_net, paddle_net):
        if hasattr(np_layer, "cell"):
            convert_params_for_cell_static(np_layer.cell, paddle_layer.cell, place)
        else:
            convert_params_for_cell_static(
                np_layer.cell_fw, paddle_layer.cell_fw, place
            )
            convert_params_for_cell_static(
                np_layer.cell_bw, paddle_layer.cell_bw, place
            )


def get_params_for_cell(np_cell, num_layers, idx):
    state = np_cell.parameters
    weight_list = [
        (f"{num_layers}.weight_{idx}", state["weight_ih"]),
        (f"{num_layers}.weight_{idx + 1}", state["weight_hh"]),
    ]
    bias_list = [
        (f"{num_layers}.bias_{idx}", state["bias_ih"]),
        (f"{num_layers}.bias_{idx + 1}", state["bias_hh"]),
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
