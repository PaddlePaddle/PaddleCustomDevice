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

import paddle_sdaa


def fused_rms_norm(x, weight, epsilion=1e-6):
    r"""
    Apply RMS Norm kernel.

    Args:
        x (Tensor): The input Tensor with data type float32.
        weight (Tensor): The weight Tensor with data type float32.
        epsilon (float): A small float number to avoid divide 0, default 1e-6.

    Returns:
        out (Tensor): The output Tensor with the same data type and shape as x.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle_sdaa

            >>> x = paddle.randn(shape=[32, 256])
            >>> weight = paddle.randn(shape=[256])
            >>> epsilon = 1e-6
            >>> out = paddle_sdaa.ops.fused_rms_norm(x, weight, epsilon)
    """
    return paddle_sdaa.custom_fused_rms_norm(x, weight, epsilion)[0]
