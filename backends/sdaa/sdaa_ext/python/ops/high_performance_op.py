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
import paddle_sdaa
from paddle.framework import in_dynamic_mode


def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    r"""
    Apply high-performance sgemmex kernel. When x(transpose_x=False) is 2D/3D Tensor and
    y(transpose_y=False) is 2D Tensor, it has better performance than paddle.matmul().

    Args:
        x (Tensor): Tensor with data type float16 or float32.
        y (Tensor): Tensor with data type float16 or float32.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        name (str, optional): Normally there is no need for user to set this parameter.
                              For detailed information, please refer to :ref:`api_guide_Name` .

    Returns:
        out (Tensor): The output Tensor with data type float32.

    Examples:
        .. code-block:: python

            import paddle
            import paddle_sdaa

            x = paddle.randn([2048, 5120], dtype='float32')
            y = paddle.randn([5120, 3456], dtype='float32')

            out = paddle_sdaa.ops.matmul(x, y)
    """
    if (
        (len(x.shape) == 3 or len(x.shape) == 2)
        and len(y.shape) == 2
        and transpose_x is False
        and transpose_y is False
    ):
        return paddle_sdaa.custom_sgemmex(x, y)[0]
    else:
        return paddle.matmul(x, y, transpose_x, transpose_y)


def linear(x, weight, bias=None, name=None):
    r"""

    Fully-connected linear transformation operator. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    If the weight is a 2-D tensor of shape :math:`[in\_features, out\_features]` ,
    input should be a multi-dimensional tensor of shape
    :math:`[batch\_size, *, in\_features]` , where :math:`*` means any number of
    additional dimensions. The linear operator multiplies input tensor with
    weight and produces an output tensor of shape :math:`[batch\_size, *, out\_features]` ,
    If :math:`bias` is not None, the bias should be a 1-D tensor of shape
    :math:`[out\_features]` and will be added to the output.

    Parameters:
        x (Tensor): Input tensor. The data type should be float16 or float32.
        weight (Tensor): Weight tensor. The data type should be float16 or float32.
        bias (Tensor, optional): Bias tensor. The data type should be float16 or float32.
                                 If it is set to None, no bias will be added to the output units.
        name (str, optional): Normally there is no need for user to set this parameter.
                              For detailed information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, the shape is :math:`[batch\_size, *, out\_features]` and the
        data type is the same with input :math:`x` .

    Examples:
        .. code-block:: python

          import paddle
          import paddle_sdaa

          x = paddle.randn((3, 2), dtype="float32")
          weight = paddle.full(shape=[2, 4], fill_value="0.5", dtype="float32", name="weight")
          bias = paddle.ones(shape=[4], dtype="float32", name="bias")

          y = paddle_sdaa.ops.linear(x, weight, bias)
    """
    if in_dynamic_mode():
        if bias is None:
            return matmul(x, weight)
        else:
            return matmul(x, weight) + bias
    else:
        raise NotImplementedError(
            "The custom linear is not support in static mode on sdaa."
        )
