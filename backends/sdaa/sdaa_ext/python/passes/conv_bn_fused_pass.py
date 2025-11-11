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

from paddle.incubate.passes import ir


@ir.RegisterPass
def custom_conv_bn_fuse_pass():
    def pattern_conv_bn(input, filter, bias, mean, scale, variance):
        conv2d = ir.PassDesc.OP.conv2d(Input=input, Filter=filter)
        bn = ir.PassDesc.OP.batch_norm(
            X=conv2d, Bias=bias, Mean=mean, Scale=scale, Variance=variance
        )
        return bn.Output("Y")

    def replace_conv_bn(input, filter, bias, mean, scale, variance):
        conv_bn = ir.PassDesc.OP.custom_fused_conv_bn(
            Input=input, Filter=filter, Scale=scale, Bias=bias, Mean=mean, Var=variance
        )

        conv_bn.Attr("strides").MappedPattern(op="conv2d", name="strides")

        conv_bn.Attr("paddings").MappedPattern(op="conv2d", name="paddings")

        conv_bn.Attr("padding_algorithm").MappedPattern(
            op="conv2d", name="padding_algorithm"
        )

        conv_bn.Attr("dilations").MappedPattern(op="conv2d", name="dilations")

        conv_bn.Attr("groups").MappedPattern(op="conv2d", name="groups")

        conv_bn.Attr("data_format").MappedPattern(op="conv2d", name="data_format")
        conv_bn.Attr("epsilon").MappedPattern(op="batch_norm", name="epsilon")
        return conv_bn

    return pattern_conv_bn, replace_conv_bn
