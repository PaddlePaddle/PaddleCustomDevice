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

import numpy as np
from test_dist_base import TestParallelDyGraphRunnerBase, runtime_main

import paddle
from paddle.base.dygraph.base import to_variable
from paddle.nn import Conv2D, SyncBatchNorm, BatchNorm


class TestLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        use_syncbn=False,
        data_format="NCHW",
        act=None,
    ):
        super().__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False,
            data_format=data_format,
        )

        if use_syncbn:
            self._sync_batch_norm = SyncBatchNorm(num_filters, data_format=data_format)
        else:
            self._sync_batch_norm = BatchNorm(num_filters, data_layout=data_format)

        self._conv2 = Conv2D(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False,
            data_format=data_format,
        )

        if use_syncbn:
            self._sync_batch_norm2 = SyncBatchNorm(num_filters, data_format=data_format)
        else:
            self._sync_batch_norm2 = BatchNorm(num_filters, data_layout=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._sync_batch_norm(y)
        y = self._conv2(y)
        y = self._sync_batch_norm2(y)
        y = paddle.nn.functional.sigmoid(y)
        y = paddle.mean(y)

        return y


class TestSyncBatchNorm(TestParallelDyGraphRunnerBase):
    def get_model(self, use_syncbn, data_format):
        model = TestLayer(1, 16, 7, use_syncbn=use_syncbn, data_format=data_format)
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(),
            batch_size=4,
            drop_last=True,
        )
        opt = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
        return model, train_reader, opt

    def run_one_loop(
        self,
        model,
        opt,
        data,
        data_format,
    ):
        batch_size = len(data)
        dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype("float32")
        img = to_variable(dy_x_data)
        if data_format == "NHWC":
            img = paddle.transpose(img, perm=[0, 2, 3, 1])

        img.stop_gradient = False

        out = model(img)

        return out


if __name__ == "__main__":
    runtime_main(TestSyncBatchNorm)
