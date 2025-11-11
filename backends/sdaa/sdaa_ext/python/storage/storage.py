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

from paddle_sdaa.sdaa_ext import *  # noqa
import paddle

storage_dict = {-1: "NCHW", 0: "CHWN"}


def tensor_storage_format(x: paddle.tensor):
    """
    Return tensor's storage format on device.

    Args:
    - x (paddle.tensor): input tensor.

    Returns:
    - string: name of the storage format.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.nn as nn
            import paddle_sdaa
            paddle.set_device("sdaa")
            x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
            conv = nn.Conv2D(4, 6, (3, 3))
            y_var = conv(x_var)
            tensor_format = paddle_sdaa.storage.tensor_storage_format(conv.weight)
            # tensor_format: "NCHW"
    """
    version = tensot_storage(x).numpy()[0]
    return storage_dict.get(version, "unknown")
