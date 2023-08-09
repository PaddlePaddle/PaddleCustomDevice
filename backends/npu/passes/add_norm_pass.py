# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_add_norm():
    def pattern(x, y, weight, bias):
        z = paddle.incubate.passes.ir.PassDesc.OP.elementwise_add(X=x, Y=y)
        layer_norm_out = paddle.incubate.passes.ir.PassDesc.OP.layer_norm(
            X=z, Scale=weight, Bias=bias
        )
        return layer_norm_out.Output("Y")

    def replace(x, y, weight, bias):
        add_norm = paddle.incubate.passes.ir.PassDesc.OP.add_norm(
            X=x, Y=y, Weight=weight, Bias=bias
        )
        add_norm.Attr("begin_norm_axis").MappedPattern(
            op="layer_norm", name="begin_norm_axis"
        )
        add_norm.Attr("epsilon").MappedPattern(op="layer_norm", name="epsilon")
        return add_norm

    return pattern, replace
