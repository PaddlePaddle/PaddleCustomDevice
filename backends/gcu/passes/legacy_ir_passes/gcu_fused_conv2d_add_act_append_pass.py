#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.incubate.passes import ir


@paddle.incubate.passes.ir.RegisterPass
def fused_conv2d_add_act_append_pass():
    def pattern(input, filter, bias):
        conv_act = ir.PassDesc.OP.fused_conv2d_add_act(
            Input=input, Filter=filter, Bias=bias
        )
        # conv_act.Attr("activation").EQ("identity")
        # conv_act.Attr("groups").EQ(1)
        act = ir.PassDesc.OP.hard_swish(X=conv_act.Output("Output"))
        return act, conv_act.Output("Outputs")

    def replace(input, filter, bias):
        conv_add_act = ir.PassDesc.OP.fused_conv2d_add_act(
            Input=input, Filter=filter, Bias=bias
        )
        conv_add_act.Attr("strides").MappedPattern(
            op="fused_conv2d_add_act", name="strides", index=0
        )
        conv_add_act.Attr("paddings").MappedPattern(
            op="fused_conv2d_add_act", name="paddings", index=0
        )
        conv_add_act.Attr("padding_algorithm").MappedPattern(
            op="fused_conv2d_add_act", name="padding_algorithm", index=0
        )
        conv_add_act.Attr("dilations").MappedPattern(
            op="fused_conv2d_add_act", name="dilations", index=0
        )
        conv_add_act.Attr("groups").MappedPattern(
            op="fused_conv2d_add_act", name="groups", index=0
        )
        conv_add_act.Attr("data_format").MappedPattern(
            op="fused_conv2d_add_act", name="data_format", index=0
        )
        conv_add_act.Attr("exhaustive_search").MappedPattern(
            op="fused_conv2d_add_act", name="exhaustive_search", index=0
        )
        conv_add_act.Attr("workspace_size_MB").MappedPattern(
            op="fused_conv2d_add_act", name="workspace_size_MB", index=0
        )
        conv_add_act.Attr("fuse_alpha").MappedPattern(
            op="fused_conv2d_add_act", name="fuse_alpha", index=0
        )
        conv_add_act.SetAttr("split_channels", [])
        conv_add_act.SetAttr("activation", "hard_swish")
        return conv_add_act.Output("Output")

    return pattern, replace
