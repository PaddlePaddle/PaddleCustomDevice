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
import unittest
import math
import os
import paddle
import paddle.nn as nn
import paddle.distributed as dist
import paddle.optimizer as opt
from paddle_sdaa.custom_parallel import DistributeAdam, DistributeMom, DistributeAdamW

paddle.seed(42)
np.random.seed(42)


class LeNet(nn.Layer):
    def __init__(self, sizes=[6, 4], layer_sizes=10):
        super(LeNet, self).__init__()
        self.amp_flag = layer_sizes % 2
        self.layer_sizes = layer_sizes
        self.amp_level = "O1"
        self._mlp_layers = []
        self._conv_layers = []
        for i in range(layer_sizes):
            linear = paddle.nn.Linear(
                in_features=sizes[0],
                out_features=sizes[1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[0])
                    )
                ),
            )
            self.add_sublayer("linear_%d" % i, linear)
            self._mlp_layers.append(linear)
        for i in range(layer_sizes):
            conv = paddle.nn.Conv2D(
                in_channels=3,
                out_channels=1,
                kernel_size=[2, 2],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[0])
                    )
                ),
            )
            self.add_sublayer("conv_%d" % i, conv)
            self._conv_layers.append(conv)

    def forward(self, inputs):
        y_dnn = 0
        input = paddle.flatten(inputs, 1, -1)
        for n_layer in self._mlp_layers:
            y_dnn += n_layer(input)
        for cv_layer in self._conv_layers:
            out = cv_layer(inputs)
            y_dnn += paddle.flatten(out, 1, -1)
        return y_dnn


train_data = []
for i in range(4):
    train_data.append(
        paddle.full(shape=[256, 3, 4, 4], fill_value=1.0, dtype=paddle.float32)
    )


def train(model, model_dist, epochs, train_dataset, optim, dist_optim):
    if hasattr(dist_optim, "rank_num"):
        np.testing.assert_allclose(dist_optim.rank_num % 32, 0)
    model_dict = {}
    model_dist_dict = {}
    name_offset = {}
    for name, param in model.named_parameters():
        # flake8: noqa
        if param.stop_gradient == False:
            # if dist_optim.rank==1 and model.layer_sizes==15:
            #     print(f'{dist_optim.rank} model_name:{name} , param.name {param.name}')
            model_dict[name] = (param.name, param)
            name_offset[name] = param.name
    for name, param in model_dist.named_parameters():
        # flake8: noqa
        if param.stop_gradient == False:
            # if dist_optim.rank==1 and model.layer_sizes==15:
            #     print(f'{dist_optim.rank} model_dist_name:{name} , param.name {param.name}')
            model_dist_dict[name] = (param.name, param)
            name_offset[param.name] = name_offset[name]

    for epoch in range(epochs):
        for batch_id, (x) in enumerate(train_dataset):
            # flake8: noqa
            if model.amp_flag == True:
                with paddle.amp.auto_cast(
                    custom_black_list={"pool2d", "relu"},  # 'elementwise_add'
                    custom_white_list={
                        "conv2d",
                        "matmul_v2",
                    },
                    level=model.amp_level,
                ):
                    predicts = model(x)
                    pred_dist = model_dist(x)
                    loss = paddle.mean(predicts)
                    loss_dist = paddle.mean(pred_dist)
                    loss.backward()
                    loss_dist.backward()
                    for name, param in model_dict.items():
                        np.testing.assert_allclose(
                            param[1].grad.numpy(), model_dist_dict[name][1].grad.numpy()
                        )
                    optim.minimize(loss)
                    dist_optim.minimize(loss_dist)
            else:
                predicts = model(x)
                pred_dist = model_dist(x)
                loss = paddle.mean(predicts)
                loss_dist = paddle.mean(pred_dist)
                loss.backward()
                loss_dist.backward()
                for name, param in model_dict.items():
                    np.testing.assert_allclose(
                        param[1].grad.numpy(), model_dist_dict[name][1].grad.numpy()
                    )
                # 更新参数
                optim.step()
                dist_optim.step()
            optim.clear_grad()
            dist_optim.clear_grad()

            for name, param in model_dict.items():
                np.testing.assert_allclose(
                    param[1].numpy(), model_dist_dict[name][1].numpy()
                )

        dist_optim._allgather_accumulators()
        for k, v in dist_optim._accumulators.items():
            for param_name, acc in v.items():
                np.testing.assert_allclose(
                    acc.numpy(), optim._accumulators[k][name_offset[param_name]].numpy()
                )


class TestDdpOptimizer(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.dtype = np.float32
        self.axis = -1
        self.set_attrs()
        self.set_dtype()
        self.set_net()

    def set_attrs(self):
        self.layer_sizes = 10
        self.opt_cls = opt.Adam
        self.ddp_opt_cls = DistributeAdam
        self.grad_clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)

    def set_dtype(self):
        self.dtype = np.float32

    def set_net(self):
        self.lenet = LeNet(sizes=[48, 9], layer_sizes=self.layer_sizes)
        lenet_dict = self.lenet.state_dict()

        self.lenet_dist = LeNet(sizes=[48, 9], layer_sizes=self.layer_sizes)
        self.lenet_dist.set_state_dict(lenet_dict)
        if self.layer_sizes % 2 == 0:
            weight_decay = paddle.regularizer.L1Decay(0.0001)
        else:
            weight_decay = 0.1

        self.opt = self.opt_cls(
            parameters=self.lenet.parameters(),
            grad_clip=self.grad_clip,
            weight_decay=weight_decay,
        )

        self.opt_dist = self.ddp_opt_cls(
            parameters=self.lenet_dist.parameters(),
            grad_clip=self.grad_clip,
            weight_decay=weight_decay,
        )


def test_class(op_type, typename, i, opt, ddp_opt, grad_clip=None):
    class TestLogDdpOptimizerMlp(TestDdpOptimizer):
        def set_attrs(self):
            self.layer_sizes = i
            self.opt_cls = opt
            self.ddp_opt_cls = ddp_opt
            self.grad_clip = grad_clip

        def set_dtype(self):
            self.dtype = typename

        def test_check_output(self):
            train(self.lenet, self.lenet_dist, 1, train_data, self.opt, self.opt_dist)

    cls_name = "{0}_{1}_1".format(op_type, typename)
    TestLogDdpOptimizerMlp.__name__ = cls_name
    globals()[cls_name] = TestLogDdpOptimizerMlp


for _typename in {np.float32}:
    clip = [
        paddle.nn.ClipGradByNorm(clip_norm=1.0),
        paddle.nn.ClipGradByValue(min=-0.5, max=0.5),
        paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
    ]
    # temp add_n is conflict to HIGH_PERFORMANCE_CONV
    if os.environ.get("HIGH_PERFORMANCE_CONV", "0") == "1":
        clip[2] = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    layer_size = int(os.environ.get("TEST_DDP_OPTIMIZER_LAYERSIZE", "0"))
    grad_clip = clip[layer_size % 3]
    test_class("TestAdam", _typename, i, opt.Adam, DistributeAdam, grad_clip=grad_clip)
    test_class(
        "TestAdamW", _typename, i, opt.AdamW, DistributeAdamW, grad_clip=grad_clip
    )
    test_class(
        "TestMom", _typename, i, opt.Momentum, DistributeMom, grad_clip=grad_clip
    )

if __name__ == "__main__":
    dist.init_parallel_env()
    unittest.main()
