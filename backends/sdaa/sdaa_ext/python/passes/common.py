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

from __future__ import print_function, division

import paddle
import paddle_sdaa

from .conv_bn_fused_pass import custom_conv_bn_fuse_pass  # noqa

from paddle.incubate.passes import ir


@ir.RegisterPass
def custom_add_n():
    def pattern(x, y, z):
        return paddle.add(paddle.add(x, y), z)

    def replace(x, y, z):
        return paddle_sdaa.custom_add_n(x, y, z)

    return pattern, replace


@ir.RegisterPass
def custom_silu_fuse_pass():
    def pattern(x):
        return x * paddle.nn.functional.sigmoid(x)

    def replace(x):
        return paddle.nn.functional.silu(x)

    return pattern, replace


# pass should be registered with input_specs
# otherwise input variable's shape will always [-1]
@ir.RegisterPass(
    input_specs={
        "input": paddle.static.InputSpec([None, 200], "float32", "input"),
        "w": paddle.static.InputSpec([200, 2], "float32", "w"),
        "bias": paddle.static.InputSpec([2], "float32", "bias"),
    }
)
def custom_fc():
    # pattern mul, elementwise_add
    # with condition x_num_col_dims = 1, y_num_col_dims = 1, axis = -1
    def pattern_fc_without_relu_1(input, w, bias):
        mul = ir.PassDesc.OP.mul(X=input, Y=w)
        mul.Attr("x_num_col_dims").EQ(1)
        mul.Attr("y_num_col_dims").EQ(1)

        elem_add = ir.PassDesc.OP.elementwise_add(X=mul.Output("Out"), Y=bias)
        elem_add.Attr("axis").EQ(-1)

        return elem_add

    def replace_fc_without_relu_1(input, w, bias):
        return paddle_sdaa.custom_fc(input, w, bias, 1, "", False)

    return pattern_fc_without_relu_1, replace_fc_without_relu_1
