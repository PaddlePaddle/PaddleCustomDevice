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

from __future__ import print_function

import numpy as np
import unittest

from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2022


class TestConv2DOp_HIGHPERFORMANCE(OpTest):
    def set_sdaa(self):
        import os

        os.environ["HIGH_PERFORMANCE_CONV"] = "1"
        self.__class__.use_custom_device = True

    def init_data_format(self):
        self.data_format = "NCHW"

    def set_optimizer(self):
        self.optimizer_name = "adam"

    def setUp(self):
        self.set_sdaa()
        self.__class__.no_need_check_grad = False
        self.__class__.op_type = "conv2d"
        self.init_data_format()
        self.init_test_case()
        self.init_amp()
        self.set_optimizer()

    def init_test_case(self):
        self.input_size = [2, 4, 8, 8]  # NHWC

    def init_amp(self):
        self.amp = False

    def test_api_dygraph(self):
        import random
        import paddle.nn as nn

        paddle.disable_static()
        paddle.device.set_device("cpu")
        paddle.seed(100)
        random.seed(100)
        np.random.seed(100)
        x = np.random.uniform(low=0, high=1.0, size=self.input_size)

        x_var_cpu = paddle.to_tensor(x, dtype="float32")
        conv_cpu = nn.Conv2DTranspose(
            4, 6, (3, 3), output_padding=1, stride=2, data_format=self.data_format
        )
        cpu_out = conv_cpu(x_var_cpu)
        weight = conv_cpu.weight.numpy()

        paddle.device.set_device("sdaa")
        x_var_sdaa = paddle.to_tensor(x, dtype="float32")
        conv_sdaa = nn.Conv2DTranspose(
            4, 6, (3, 3), output_padding=1, stride=2, data_format=self.data_format
        )

        conv_sdaa.weight.set_value(paddle.to_tensor(weight))
        if self.amp:
            with paddle.amp.auto_cast(custom_white_list={"conv2d"}, level="O1"):
                cpu_s = conv_sdaa(x_var_sdaa)
        else:
            cpu_s = conv_sdaa(x_var_sdaa)

        if self.amp:
            atol = 1e-2
        else:
            atol = 1e-2
        np.testing.assert_allclose(cpu_s.numpy(), cpu_out.numpy(), atol=atol)

    def test_backward_api_dygraph(self):
        import paddle.nn as nn

        x = np.random.uniform(low=0, high=1.0, size=self.input_size)
        x_var = paddle.to_tensor(x, dtype="float32")
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        conv = nn.Conv2DTranspose(
            4, 6, (3, 3), output_padding=1, stride=2, data_format=self.data_format
        )
        input_n = x_var.numpy()
        weight = conv.weight.numpy()
        bias = conv.bias.numpy()

        paddle.device.set_device("sdaa")
        conv_layer = nn.Conv2DTranspose(
            4, 6, (3, 3), output_padding=1, stride=2, data_format=self.data_format
        )
        x_var = paddle.to_tensor(input_n)
        conv_layer.weight.set_value(paddle.to_tensor(weight))
        conv_layer.bias.set_value(paddle.to_tensor(bias))
        if self.optimizer_name == "adam":
            opti = paddle.optimizer.Adam(
                learning_rate=0.1, parameters=conv_layer.parameters(), weight_decay=0.01
            )
        elif self.optimizer_name == "momentum":
            opti = paddle.optimizer.Momentum(
                learning_rate=0.1, parameters=conv_layer.parameters(), weight_decay=0.01
            )
        loss_l = []
        for i in range(10):
            if self.amp:
                with paddle.amp.auto_cast(custom_white_list={"conv2d"}, level="O1"):
                    y_var = conv_layer(x_var)
                    loss = paddle.mean(y_var)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(opti)
                scaler.update()
                opti.clear_grad(set_to_zero=False)
                loss_l.append(loss.item())
            else:
                y_var = conv_layer(x_var)
                loss = paddle.mean(y_var)
                loss.backward()
                opti.step()
                opti.clear_grad()
                loss_l.append(loss.item())

        paddle.device.set_device("cpu")
        x_var = paddle.to_tensor(input_n)
        x_var.stop_gradient = False
        conv1 = nn.Conv2DTranspose(
            4, 6, (3, 3), output_padding=1, stride=2, data_format=self.data_format
        )
        conv1.weight.set_value(paddle.to_tensor(weight))
        conv1.bias.set_value(paddle.to_tensor(bias))
        if self.optimizer_name == "adam":
            optmi1 = paddle.optimizer.Adam(
                learning_rate=0.1, parameters=conv1.parameters(), weight_decay=0.01
            )
        elif self.optimizer_name == "momentum":
            optmi1 = paddle.optimizer.Momentum(
                learning_rate=0.1, parameters=conv1.parameters(), weight_decay=0.01
            )
        loss_c = []
        for i in range(10):
            y_var = conv1(x_var)
            loss = paddle.mean(y_var)
            loss.backward()
            optmi1.step()
            optmi1.clear_grad()
            loss_c.append(loss.item())
        np.testing.assert_allclose(np.array(loss_c), np.array(loss_l), atol=1e-2)


class TestConv2DOp_HIGHPERFORMANCE_adam_amp(TestConv2DOp_HIGHPERFORMANCE):
    def init_amp(self):
        self.amp = True


class TestConv2DOp_HIGHPERFORMANCE_adam_NCHW(TestConv2DOp_HIGHPERFORMANCE):
    def init_test_case(self):
        self.input_size = [2, 8, 8, 4]  # NCHW

    def init_data_format(self):
        self.data_format = "NHWC"


class TestConv2DOp_HIGHPERFORMANCE_momentum(TestConv2DOp_HIGHPERFORMANCE):
    def set_optimizer(self):
        self.optimizer_name = "momentum"


class TestConv2DOp_HIGHPERFORMANCE_momentum_amp(TestConv2DOp_HIGHPERFORMANCE):
    def set_optimizer(self):
        self.optimizer_name = "momentum"

    def init_amp(self):
        self.amp = True


class TestConv2DOp_HIGHPERFORMANCE_momentum_amp_NCHW(TestConv2DOp_HIGHPERFORMANCE):
    def init_test_case(self):
        self.input_size = [2, 4, 8, 8]  # NCHW

    def set_optimizer(self):
        self.optimizer_name = "momentum"

    def init_amp(self):
        self.amp = True

    def init_data_format(self):
        self.data_format = "NCHW"


if __name__ == "__main__":
    unittest.main()
