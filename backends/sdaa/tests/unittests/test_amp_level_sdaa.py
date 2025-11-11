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

import unittest

from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2022


class Test_Amp_o1_adamW(OpTest):
    def set_sdaa(self):
        pass

        self.__class__.use_custom_device = True

    def init_data_format(self):
        self.data_format = "NHWC"

    def set_optimizer(self):
        self.optimizer_name = "adamW"

    def set_multi_precision(self):
        self.multi_precision = False

    def setUp(self):
        self.set_sdaa()
        self.__class__.no_need_check_grad = False
        self.__class__.op_type = "conv2d"
        self.init_data_format()
        self.init_test_case()
        self.init_amp()
        self.set_optimizer()
        self.set_multi_precision()

    def init_test_case(self):
        self.input_size = [4, 16, 16, 3]  # NHWC

    def init_amp(self):
        self.amp = "O1"

    def test_api_dygraph(self):
        import random
        import numpy as np
        import paddle.nn as nn
        import paddle

        paddle.seed(100)
        random.seed(100)
        np.random.seed(100)
        paddle.disable_static()

        x = np.random.uniform(low=0, high=1.0, size=self.input_size)
        x_var = paddle.to_tensor(x, dtype="float32")

        conv = nn.Conv2D(3, 16, (3, 3), data_format=self.data_format)
        input_n = x_var.numpy()
        weight = conv.weight.numpy()
        bias = conv.bias.numpy()

        paddle.device.set_device("sdaa")
        conv_layer = nn.Conv2D(3, 16, (3, 3), data_format=self.data_format)
        x_var = paddle.to_tensor(input_n)
        conv_layer.weight.set_value(paddle.to_tensor(weight))
        conv_layer.bias.set_value(paddle.to_tensor(bias))
        if self.optimizer_name == "adam":
            opti = paddle.optimizer.Adam(
                learning_rate=0.1,
                parameters=conv_layer.parameters(),
                weight_decay=0.01,
                multi_precision=self.multi_precision,
            )
        if self.optimizer_name == "adamW":
            opti = paddle.optimizer.AdamW(
                learning_rate=0.1,
                parameters=conv_layer.parameters(),
                weight_decay=0.01,
                multi_precision=self.multi_precision,
            )
        model = paddle.amp.decorate(models=conv_layer, level=self.amp)

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        loss_l = []
        for i in range(5):

            with paddle.amp.auto_cast(custom_white_list={"conv2d"}, level=self.amp):
                y_var = model(x_var)
                loss = paddle.mean(y_var)
            scaled = scaler.scale(loss)  # loss 缩放，乘以系数 loss_scaling
            scaled.backward()  # 反向传播
            scaler.step(opti)  # 更新参数（参数梯度先除系数 loss_scaling 再更新参数）
            scaler.update()  # 基于动态 loss_scaling 策略更新 loss_scaling 系数
            opti.clear_grad(set_to_zero=False)
            # 记录训练 loss 及训练时长
            train_loss = loss.numpy()
            loss_l.append(train_loss)

        paddle.device.set_device("cpu")
        conv_layer_cpu = nn.Conv2D(3, 16, (3, 3), data_format=self.data_format)
        x_var = paddle.to_tensor(input_n)
        conv_layer_cpu.weight.set_value(paddle.to_tensor(weight))
        conv_layer_cpu.bias.set_value(paddle.to_tensor(bias))
        if self.optimizer_name == "adam":
            opti_cpu = paddle.optimizer.Adam(
                learning_rate=0.1,
                parameters=conv_layer_cpu.parameters(),
                weight_decay=0.01,
                multi_precision=True,
            )
        if self.optimizer_name == "adamW":
            opti_cpu = paddle.optimizer.AdamW(
                learning_rate=0.1,
                parameters=conv_layer_cpu.parameters(),
                weight_decay=0.01,
                multi_precision=True,
            )
        model_cpu = paddle.amp.decorate(models=conv_layer_cpu, level="O1")

        scaler_cpu = paddle.amp.GradScaler(init_loss_scaling=1024)
        loss_c = []
        for i in range(5):
            with paddle.amp.auto_cast(custom_white_list={"conv2d"}, level="O1"):
                y_var = model_cpu(x_var)
                loss = paddle.mean(y_var)
            scaled = scaler_cpu.scale(loss)
            scaled.backward()
            scaler_cpu.step(opti_cpu)
            scaler_cpu.update()
            opti_cpu.clear_grad(set_to_zero=False)
            train_loss = loss.numpy()
            loss_c.append(train_loss)

        np.testing.assert_allclose(np.array(loss_c), np.array(loss_l), atol=1e-2)


class Test_Amp_o2_adamW_01(Test_Amp_o1_adamW):
    def set_optimizer(self):
        self.optimizer_name = "adamW"

    def init_amp(self):
        self.amp = "O2"

    def set_multi_precision(self):
        self.multi_precision = False


class Test_Amp_o2_adamW_02(Test_Amp_o1_adamW):
    def set_optimizer(self):
        self.optimizer_name = "adamW"

    def init_amp(self):
        self.amp = "O2"

    def set_multi_precision(self):
        self.multi_precision = True


class Test_Amp_o1_adam_01(Test_Amp_o1_adamW):
    def set_optimizer(self):
        self.optimizer_name = "adam"


class Test_Amp_o2_adam_02(Test_Amp_o1_adamW):
    def set_optimizer(self):
        self.optimizer_name = "adam"

    def init_amp(self):
        self.amp = "O2"

    def set_multi_precision(self):
        self.multi_precision = False


class Test_Amp_o2_adam_02(Test_Amp_o1_adamW):
    def set_optimizer(self):
        self.optimizer_name = "adam"

    def init_amp(self):
        self.amp = "O2"

    def set_multi_precision(self):
        self.multi_precision = True


if __name__ == "__main__":
    unittest.main()
