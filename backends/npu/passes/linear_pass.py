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
def generate_linear():
    def pattern(x, weight, bias):
        matmul = paddle.incubate.passes.ir.PassDesc.OP.matmul_v2(X=x, Y=weight)
        return paddle.incubate.passes.ir.PassDesc.OP.elementwise_add(X=matmul, Y=bias)

    def replace(x, weight, bias):
        return paddle.incubate.passes.ir.PassDesc.OP.linear(
            Input=x, Weight=weight, Bias=bias
        )

    return pattern, replace
