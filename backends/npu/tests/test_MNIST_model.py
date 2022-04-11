#--encoding=utf-8

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# 1 定义数据读取器：读取数据和预处理操作。
# 2 定义模型和优化器：搭建神经网络结构。
# 3 训练：配置优化器、学习率、训练参数。循环调用训练过程，循环执行“前向计算 + 损失函数 + 反向传播”。
# 4 评估测试：将训练好的模型保存并评估测试。

import numpy as np
import paddle
from paddle.nn import Conv2D, MaxPool2D, ReLU
from paddle.optimizer import Adam

# 定义批大小
BATCH_SIZE = 64


# 1 定义数据读取器：读取数据和预处理操作。
# 由于paddle.io.DataLoader只接受numpy ndarray或者paddle Tensor作为数据输入
# 该自定义类将MNIST数据reshape并转化为numpy ndarray类型，并且将数据从[0, 255] 转化到 [-1, 1]
class MnistDataset(paddle.vision.datasets.MNIST):
    def __init__(self, mode, return_label=True):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        img = img / 255.0 * 2.0 - 1.0
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int')
        return img,

    def __len__(self):
        return len(self.images)


# 通过调用paddle.io.DataLoader来构造reader
train_reader = paddle.io.DataLoader(
    MnistDataset(mode='train'), batch_size=BATCH_SIZE, drop_last=True)
test_reader = paddle.io.DataLoader(
    MnistDataset(mode='test'), batch_size=BATCH_SIZE, drop_last=True)


# 定义MNIST网络，必须继承自paddle.nn.Layer
# 该网络由两个SimpleImgConvPool子网络、reshape层、matmul层、softmax层、accuracy层组成
class MNIST(paddle.nn.Layer):
    # 在__init__构造函数中会执行变量的初始化、参数初始化、子网络初始化的操作
    # 本例中执行了self.shape变量、matmul层中参数self.output_weight、SimpleImgConvPool子网络的初始化操作
    # 并且定义了衡量输出准确率的accuracy的paddle.metric.Accuracy
    def __init__(self):
        super(MNIST, self).__init__()
        self.shape = 1 * 28 * 28
        self.size = 10
        # 定义全连接层的参数
        self.output_weight = self.create_parameter([self.shape, self.size])

        # 定义计算accuracy的层
        self.accuracy = paddle.metric.Accuracy()

    # forward函数实现了MNIST网络的执行逻辑
    def forward(self, inputs, label=None):
        x = paddle.reshape(inputs, shape=[-1, self.shape])
        x = paddle.matmul(x, self.output_weight)
        x = paddle.nn.functional.softmax(x)
        if label is not None:
            # Reset只返回当前batch的准确率
            self.accuracy.reset()
            correct = self.accuracy.compute(x, label)
            self.accuracy.update(correct)
            acc = self.accuracy.accumulate()
            return x, acc
        else:
            return x


paddle.set_device('ascend')
# 3 训练：配置优化器、学习率、训练参数。循环调用训练过程，循环执行“前向计算 + 损失函数 + 反向传播”。
# 定义MNIST类的对象
mnist = MNIST()
# 定义优化器为SGD，学习旅learning_rate为0.001
# 注意动态图模式下必须传入parameters参数，该参数为需要优化的网络参数，本例需要优化mnist网络中的所有参数
adam = Adam(learning_rate=0.001, parameters=mnist.parameters())

# 设置全部样本的训练次数
epoch_num = 1
# 执行epoch_num次训练
for epoch in range(epoch_num):
    # 读取训练数据进行训练
    for batch_id, data in enumerate(train_reader()):
        # train_reader 返回的是img和label已经是Tensor类型，可以动态图使用
        img = data[0]
        label = data[1]

        # 网络正向执行
        pred, acc = mnist(img, label)
        # 计算损失值
        avg_loss = paddle.nn.functional.cross_entropy(pred, label)
        #avg_loss = paddle.mean(loss)
        #avg_loss = loss.mean()
        # 执行反向计算
        avg_loss.backward()
        # 参数更新
        adam.step()
        # 将本次计算的梯度值清零，以便进行下一次迭代和梯度更新
        adam.clear_grad()

        # 输出对应epoch、batch_id下的损失值，预测精确度
        if batch_id % 100 == 0:
            print("Epoch {} step {}, Loss = {:}, Accuracy = {:}".format(
                epoch, batch_id, avg_loss.numpy(), acc))
# 保存训练好的模型
model_dict = mnist.state_dict()
paddle.save(model_dict, "mnist.pdparams")
