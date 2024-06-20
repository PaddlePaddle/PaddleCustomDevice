#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest

from op_test import OpTest
import paddle

import os
import random
import paddle.nn as nn

from paddle.base import dygraph

from paddle.optimizer import lr, AdamW

paddle.enable_static()
SEED = 2022

os.environ["HIGH_PERFORMANCE_CONV"] = "1"


class TestConv2DOp_HIGHPERFORMANCE(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_data_format(self):
        self.data_format = "NHWC"

    def set_optimizer(self):
        self.optimizer_name = "adam"
        self.use_multi_tensor = False

    def setUp(self):
        self.set_sdaa()
        self.__class__.no_need_check_grad = False
        self.__class__.op_type = "conv2d"
        self.init_data_format()
        self.init_test_case()
        self.init_amp()
        self.set_optimizer()

    def init_test_case(self):
        self.input_size = [4, 16, 16, 3]  # NHWC

    def init_amp(self):
        self.amp = False

    def test_api_dygraph(self):
        import paddle.nn as nn

        paddle.disable_static()
        paddle.device.set_device("cpu")
        paddle.seed(100)
        random.seed(100)
        np.random.seed(100)
        x = np.random.uniform(low=0, high=1.0, size=self.input_size)

        x_var_cpu = paddle.to_tensor(x, dtype="float32")
        conv_cpu = nn.Conv2D(3, 16, (3, 3), data_format=self.data_format)
        cpu_out = conv_cpu(x_var_cpu)
        weight = conv_cpu.weight.numpy()

        paddle.device.set_device("sdaa")
        x_var_sdaa = paddle.to_tensor(x, dtype="float32")
        conv_sdaa = nn.Conv2D(3, 16, (3, 3), data_format=self.data_format)

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
        x = np.random.uniform(low=0, high=1.0, size=self.input_size)
        x_var = paddle.to_tensor(x, dtype="float32")
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
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
                learning_rate=0.1, parameters=conv_layer.parameters(), weight_decay=0.01
            )
        elif self.optimizer_name == "momentum":
            opti = paddle.optimizer.Momentum(
                learning_rate=0.1,
                parameters=conv_layer.parameters(),
                weight_decay=0.01,
                use_multi_tensor=self.use_multi_tensor,
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
        conv1 = nn.Conv2D(3, 16, (3, 3), data_format=self.data_format)
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
        self.input_size = [4, 3, 16, 16]  # NCHW

    def init_data_format(self):
        self.data_format = "NCHW"


class TestConv2DOp_HIGHPERFORMANCE_momentum(TestConv2DOp_HIGHPERFORMANCE):
    def set_optimizer(self):
        self.optimizer_name = "momentum"
        self.use_multi_tensor = False


class TestConv2DOp_HIGHPERFORMANCE_momentum_amp(TestConv2DOp_HIGHPERFORMANCE):
    def set_optimizer(self):
        self.optimizer_name = "momentum"
        self.use_multi_tensor = False

    def init_amp(self):
        self.amp = True


class TestConv2DOp_HIGHPERFORMANCE_momentum_amp_NCHW(TestConv2DOp_HIGHPERFORMANCE):
    def init_test_case(self):
        self.input_size = [4, 3, 16, 16]  # NCHW

    def set_optimizer(self):
        self.optimizer_name = "momentum"
        self.use_multi_tensor = False

    def init_amp(self):
        self.amp = True

    def init_data_format(self):
        self.data_format = "NCHW"


class TestConv2DOp_HIGHPERFORMANCE_merged_momentum(TestConv2DOp_HIGHPERFORMANCE):
    def set_optimizer(self):
        self.optimizer_name = "momentum"
        self.use_multi_tensor = True


class TestConv2DOp_HIGHPERFORMANCE(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_data_format(self):
        self.data_format = "NHWC"

    def setUp(self):
        self.set_sdaa()
        self.__class__.no_need_check_grad = False
        self.__class__.op_type = "conv2d"
        self.init_data_format()
        self.init_test_case()
        self.init_amp()

    def init_test_case(self):
        self.input_size = [64, 224, 224, 3]  # NHWC

    def init_amp(self):
        self.amp = False

    def test_api_dygraph(self):
        paddle.disable_static()
        paddle.device.set_device("cpu")
        paddle.seed(100)
        random.seed(100)
        np.random.seed(100)
        x = np.random.uniform(low=0, high=1.0, size=self.input_size)

        x_var_cpu = paddle.to_tensor(x, dtype="float32")
        conv_cpu = nn.Conv2D(3, 64, (7, 7), data_format=self.data_format)
        cpu_out = conv_cpu(x_var_cpu)
        weight = conv_cpu.weight.numpy()

        paddle.device.set_device("sdaa")
        x_var_sdaa = paddle.to_tensor(x, dtype="float32")
        conv_sdaa = nn.Conv2D(3, 64, (7, 7), data_format=self.data_format)

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
        x = np.random.uniform(low=0, high=1.0, size=self.input_size)
        x_var = paddle.to_tensor(x, dtype="float32")
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        conv = nn.Conv2D(3, 64, (7, 7), data_format=self.data_format)
        input_n = x_var.numpy()
        weight = conv.weight.numpy()
        bias = conv.bias.numpy()

        paddle.device.set_device("sdaa")
        conv_layer = nn.Conv2D(3, 64, (7, 7), data_format=self.data_format)
        x_var = paddle.to_tensor(input_n)
        conv_layer.weight.set_value(paddle.to_tensor(weight))
        conv_layer.bias.set_value(paddle.to_tensor(bias))
        momentum = paddle.optimizer.Momentum(
            learning_rate=0.1, parameters=conv_layer.parameters(), weight_decay=0.01
        )
        loss_l = []
        for i in range(1):
            if self.amp:
                with paddle.amp.auto_cast(custom_white_list={"conv2d"}, level="O1"):
                    y_var = conv_layer(x_var)
                    loss = paddle.mean(y_var)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(momentum)
                scaler.update()
                momentum.clear_grad(set_to_zero=False)
                loss_l.append(loss.item())
            else:
                y_var = conv_layer(x_var)
                loss = paddle.mean(y_var)
                loss.backward()
                momentum.step()
                momentum.clear_grad()
                loss_l.append(loss.item())

        paddle.device.set_device("cpu")
        x_var = paddle.to_tensor(input_n)
        x_var.stop_gradient = False
        conv1 = nn.Conv2D(3, 64, (7, 7), data_format=self.data_format)
        conv1.weight.set_value(paddle.to_tensor(weight))
        conv1.bias.set_value(paddle.to_tensor(bias))
        momentum1 = paddle.optimizer.Momentum(
            learning_rate=0.1, parameters=conv1.parameters(), weight_decay=0.01
        )
        loss_c = []
        for i in range(1):
            y_var = conv1(x_var)
            loss = paddle.mean(y_var)
            loss.backward()
            momentum1.step()
            momentum1.clear_grad()
            loss_c.append(loss.item())

        np.testing.assert_allclose(np.array(loss_c), np.array(loss_l), atol=1e-2)


class TestConv2DOp_HIGHPERFORMANCE_amp(TestConv2DOp_HIGHPERFORMANCE):
    def init_amp(self):
        self.amp = True


class TestConv2DOp_HIGHPERFORMANCE_NCHW(TestConv2DOp_HIGHPERFORMANCE):
    def init_test_case(self):
        self.input_size = [64, 3, 224, 224]  # NCHW

    def init_data_format(self):
        self.data_format = "NCHW"


def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_sample, input_shape):
        self.num_sample = num_sample
        self.input_shape = input_shape

    def __getitem__(self, idx):
        img = np.ones(shape=self.input_shape).astype("float32")
        label = np.ones(1).astype("int64")
        return img, label

    def __len__(self):
        return self.num_sample


class TestConv2dWithHIGHPERFORMANCE(unittest.TestCase):
    def setUp(self):
        self.seed = 2022
        self.num_sample = 12

        self.init_test_case()
        self.init_amp()

    def get_loss_dygraph_api_with_place(self, place):
        with dygraph.guard(place):
            set_seed(self.seed)
            self.train_loader = paddle.io.DataLoader(
                RandomDataset(
                    num_sample=self.num_sample, input_shape=self.input_shape[1:]
                ),
                batch_size=self.input_shape[0],
                shuffle=False,
                drop_last=True,
                num_workers=0,
            )
            self.lr = self.build_lr()
            self.model = self.build_model()
            self.optimizer = self.build_optimizer()
            losses = []
            if self.use_amp:
                scaler = paddle.amp.GradScaler()
            for _, data in enumerate(self.train_loader()):
                img, _ = data
                if self.use_amp:
                    with paddle.amp.auto_cast():
                        # model forward
                        loss = self.model(img)
                    # model backward
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    scaler.minimize(self.optimizer, scaled_loss)
                else:
                    # model forward
                    loss = self.model(img)
                    # model backward
                    loss.backward()
                    self.optimizer.step()
                self.lr.step()

                losses.append(loss.numpy())
        return losses

    def test_dygraph_api(self):
        cpu_loss = self.get_loss_dygraph_api_with_place(paddle.CPUPlace())
        sdaa_loss = self.get_loss_dygraph_api_with_place(paddle.CustomPlace("sdaa", 0))

        for i in range(len(cpu_loss)):
            np.testing.assert_allclose(cpu_loss[i], sdaa_loss[i], atol=1e-4)

    def init_amp(self):
        self.use_amp = True

    def init_test_case(self):
        self.input_shape = [4, 16, 16, 3]

    def build_lr(self):
        decay_lr = lr.PiecewiseDecay([1], [0.001, 0.02])
        return decay_lr

    def build_optimizer(self):
        grad_clip = nn.ClipGradByGlobalNorm(clip_norm=0.1)
        optimizer = AdamW(
            weight_decay=0.01, parameters=self.model.parameters(), grad_clip=grad_clip
        )
        return optimizer

    def build_model(self):
        class ConvModel(nn.Layer):
            def __init__(self, name_scope=None, dtype="float32"):
                super().__init__(name_scope, dtype)
                self.conv = nn.Conv2D(3, 64, (7, 7), data_format="NHWC")

            def forward(self, *inputs, **kwargs):
                data = inputs[0]
                p = self.conv(data)
                return paddle.mean(p)

        model = ConvModel()
        return model


class TestConv2dWithMultiply(TestConv2dWithHIGHPERFORMANCE):
    def init_test_case(self):
        self.input_shape = [4, 16, 16, 3]

        self.y_shape = [1]

    def test_multiply(self):
        place = paddle.CustomPlace("sdaa", 0)
        np_y = np.random.randn(*self.y_shape).astype("float32")

        self.get_loss_dygraph_api_with_place(place)

        with dygraph.guard(place):
            y = paddle.to_tensor(np_y)
            for grad in self.model.parameters():
                if grad is None:
                    continue

                out = paddle.multiply(grad, y)
                np_out = np.multiply(grad.numpy(), np_y)
                np.testing.assert_allclose(np_out, out.numpy())

                self.assertRaises(
                    ValueError,
                    paddle.multiply,
                    grad,
                    paddle.to_tensor([1, 2], dtype=grad.dtype),
                )


if __name__ == "__main__":
    unittest.main()
