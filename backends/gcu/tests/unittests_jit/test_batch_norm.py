# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from api_base import ApiBase
import paddle
import random
import pytest
import numpy as np
from paddle.io import Dataset

paddle.seed(33)
random.seed(33)
np.random.seed(33)

CLASS_NUM = 10
TRAIN_SAMPLE_NUM = 4
EVAL_SAMPLE_NUM = 4
BS = 4
CHANNEL_NUM = 4
FEATURE_SIZE = 64  # 8的整数倍


class MyDataset(Dataset):
    def __init__(self, mode="train", bn_type="bn2d"):
        super().__init__()
        self.mode = mode
        self.bn_type = bn_type

    def __getitem__(self, index):
        if self.bn_type == "bn1d":
            image = (
                np.random.random(
                    size=(int(CHANNEL_NUM), int(FEATURE_SIZE / CHANNEL_NUM))
                ).astype("float32")
                * 255
            )
        elif self.bn_type == "bn2d":
            image = (
                np.random.random(
                    size=(
                        int(CHANNEL_NUM),
                        int(FEATURE_SIZE / CHANNEL_NUM / 4),
                        int(FEATURE_SIZE / CHANNEL_NUM / 4),
                    )
                ).astype("float32")
                * 255
            )
        else:
            image = (
                np.random.random(
                    size=(
                        int(CHANNEL_NUM),
                        int(FEATURE_SIZE / CHANNEL_NUM / 4),
                        int(FEATURE_SIZE / CHANNEL_NUM / 8),
                        int(FEATURE_SIZE / CHANNEL_NUM / 8),
                    )
                ).astype("float32")
                * 255
            )
        label = np.random.randint(0, CLASS_NUM, (1)).astype("int64")
        return image, label

    def __len__(self):
        if self.mode == "train":
            return TRAIN_SAMPLE_NUM
        else:
            return EVAL_SAMPLE_NUM


class SimplifyNet(paddle.nn.Layer):
    def __init__(self, bn_type):
        super(SimplifyNet, self).__init__()
        if bn_type == "bn1d":
            self.bn = paddle.nn.BatchNorm1D(CHANNEL_NUM)
            self.linear = paddle.nn.Linear(FEATURE_SIZE, 4)
        elif bn_type == "bn2d":
            self.bn = paddle.nn.BatchNorm2D(CHANNEL_NUM)
            self.linear = paddle.nn.Linear(FEATURE_SIZE, 4)
        else:
            self.bn = paddle.nn.BatchNorm3D(CHANNEL_NUM)
            self.linear = paddle.nn.Linear(FEATURE_SIZE, 4)

    # @paddle.jit.to_static
    def forward(self, x):
        x = self.bn(x)
        x = paddle.reshape(x, [BS, FEATURE_SIZE])
        x = self.linear(x)
        return x


def train(model, loader):
    model.train()
    loss = -1
    for epoch_id in range(1):
        for batch_id, (image, label) in enumerate(loader()):
            out = model(image)

            loss = paddle.mean(out - label)
            opt = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=model.parameters()
            )
            loss.backward()
            opt.step()
            opt.clear_grad()
            loss = np.mean(float(loss))
            print(
                "Epoch {} batch {}: loss = {}".format(epoch_id, batch_id, np.mean(loss))
            )
    return loss


def eval(model, loader):
    model.eval()
    ret = -100
    for batch_id, (image, label) in enumerate(loader()):
        out = model(image)
        loss = paddle.mean(out - label)
        ret = np.mean(float(loss))
    return ret


def RunBatchNorm(device, type):
    paddle.set_device(device)
    train_dataset = MyDataset(mode="train", bn_type=type)
    eval_dataset = MyDataset(mode="test", bn_type=type)

    train_loader = paddle.io.DataLoader(
        train_dataset, batch_size=BS, shuffle=True, drop_last=True, num_workers=0
    )

    eval_loader = paddle.io.DataLoader(
        eval_dataset, batch_size=BS, shuffle=False, drop_last=False, num_workers=0
    )
    model = SimplifyNet(type)
    train(model, train_loader)
    return eval(model, eval_loader)


# eager mode test
@pytest.mark.batch_norm
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_norm1d():
    gcu_eval_value = RunBatchNorm("gcu", "bn1d")
    cpu_eval_value = RunBatchNorm("cpu", "bn1d")
    np.allclose(cpu_eval_value, gcu_eval_value, atol=1e-5, rtol=1e-5)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_norm2d():
    gcu_eval_value = RunBatchNorm("gcu", "bn2d")
    cpu_eval_value = RunBatchNorm("cpu", "bn2d")
    np.allclose(cpu_eval_value, gcu_eval_value, atol=1e-5, rtol=1e-5)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_norm3d():
    gcu_eval_value = RunBatchNorm("gcu", "bn3d")
    cpu_eval_value = RunBatchNorm("cpu", "bn3d")
    np.allclose(cpu_eval_value, gcu_eval_value, atol=1e-5, rtol=1e-5)


test1 = ApiBase(
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=["data"],
    feed_shapes=[[2, 3, 2, 4]],
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_norm_1():
    data = np.random.random(size=[2, 3, 2, 4]).astype("float32")
    test1.run(feed=[data])


# static mode test
test2 = ApiBase(
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=["data"],
    feed_shapes=[[1, 5, 7, 9]],
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_norm_2():
    data = np.random.random(size=[1, 5, 7, 9]).astype("float32")
    test2.run(feed=[data])


test3 = ApiBase(
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=["data"],
    feed_shapes=[[1, 8, 352, 640]],
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_norm_3():
    data = np.random.uniform(-1, 1, (1, 8, 352, 640)).astype("float32")
    test3.run(feed=[data])


test4 = ApiBase(
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=["data"],
    feed_shapes=[[8, 3, 224, 224]],
    is_train=False,
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_norm_4():
    data = np.random.uniform(-1, 1, (8, 3, 224, 224)).astype("float32")
    test4.run(feed=[data], is_test=True)
