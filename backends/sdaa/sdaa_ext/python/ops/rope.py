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


def fused_rotary_position_embedding(query, key, cos, sin):
    r"""
    Apply Rotary Position Embedding kernel.

    Args:
        query (Tensor): The query Tensor with data type float32. The shape of q must be [seq_len, batch_size, num_heads, head_dim] and head_dim must be a multiple of 32.
        key (Tensor): The key Tensor with data type float32. The shape of k must be [seq_len, batch_size, num_heads, head_dim] and head_dim must be a multiple of 32.
        cos  (Tensor): The cos Tensor with data type float32.
        sin (Tensor): The sin Tensor with data type float32.

    Returns:
        out_query (Tensor): The output Tensor with the same data type and shape as query.
        out_key (Tensor): The output Tensor with the same data type and shape as key.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle_sdaa

            >>> q = paddle.randn([1, 1, 4, 64], dtype='float32')
            >>> k = paddle.randn([1, 1, 4, 64], dtype='float32')

            >>> x = paddle.randn([1, 1, 1, 64], dtype='float32')
            >>> y = paddle.randn([1, 1, 1, 64], dtype='float32')
            >>> sin = paddle.sin(x)
            >>> cos = paddle.cos(y)
            >>> out_q, out_k = paddle_sdaa.ops.fused_rotary_position_embedding(q, k, cos, sin)
    """
    return paddle_sdaa.custom_fused_rotary_position_embedding(query, key, cos, sin)
