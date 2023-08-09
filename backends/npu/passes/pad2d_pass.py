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
def generate_pad2d():
    def pattern(x):
        pad3d_in = paddle.incubate.passes.ir.PassDesc.OP.unsqueeze2(X=x)
        pad3d_out = paddle.incubate.passes.ir.PassDesc.OP.pad3d(
            X=pad3d_in.Output("Out")
        )
        res = paddle.incubate.passes.ir.PassDesc.OP.squeeze2(X=pad3d_out)
        return res.Output("Out")

    def replace(x):
        return paddle.incubate.passes.ir.PassDesc.OP.pad2d(X=x)

    return pattern, replace
