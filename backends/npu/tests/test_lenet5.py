#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
import argparse
import time

EPOCH_NUM = 1
BATCH_SIZE = 4096


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'ascend'],
        default="ascend",
        help='device type, support cpu and ascend')
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp32', 'ampo1', 'ampo2'],
        default="fp32",
        help='precision type, support fp32 ampo1 and ampo2')
    parser.add_argument(
        '--to_static',
        type=str2bool,
        default=False,
        help='whether to enable dynamic to static or not, true or false')
    return parser.parse_args()


class LeNet5(nn.Layer):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2D(num_features=6),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=400, out_features=120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = paddle.flatten(out, 1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def train_func_base(epoch_id, train_loader, model, cost, optimizer):
    total_step = len(train_loader)
    epoch_start = time.time()
    for batch_id, (images, labels) in enumerate(train_loader()):
        # forward
        outputs = model(images)
        loss = cost(outputs, labels)
        # backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        print("Epoch [{}/{}], Step [{}/{}], Loss: {}".format(
            epoch_id + 1, EPOCH_NUM, batch_id + 1, total_step, loss.numpy()))
    epoch_end = time.time()
    print(
        f"Epoch ID: {epoch_id+1}, FP32 train epoch time: {(epoch_end - epoch_start) * 1000} ms"
    )


def train_func_ampo1(epoch_id, train_loader, model, cost, optimizer, scaler):
    total_step = len(train_loader)
    epoch_start = time.time()
    for batch_id, (images, labels) in enumerate(train_loader()):
        # forward
        with paddle.amp.auto_cast(level='O1'):
            outputs = model(images)
            loss = cost(outputs, labels)
        # backward and optimize
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()
        print("Epoch [{}/{}], Step [{}/{}], Loss: {}".format(
            epoch_id + 1, EPOCH_NUM, batch_id + 1, total_step, loss.numpy()))
    epoch_end = time.time()
    print(
        f"Epoch ID: {epoch_id+1}, AMPO1 train epoch time: {(epoch_end - epoch_start) * 1000} ms"
    )


def test_func(epoch_id, test_loader, model, cost):
    # evaluation every epoch finish
    model.eval()
    avg_acc = [[], []]
    for batch_id, (images, labels) in enumerate(test_loader()):
        # forward
        outputs = model(images)
        loss = cost(outputs, labels)
        # accuracy
        acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
        avg_acc[0].append(acc_top1.numpy())
        avg_acc[1].append(acc_top5.numpy())
    model.train()
    print(
        f"Epoch ID: {epoch_id+1}, Top1 accurary: {np.array(avg_acc[0]).mean()}, Top5 accurary: {np.array(avg_acc[1]).mean()}"
    )


def infer_func(saved_model, device_type=None):
    # create config
    config = paddle.inference.Config(saved_model + '.pdmodel',
                                     saved_model + '.pdiparams')
    # enable custom device
    if device_type == "ascend":
        config.enable_custom_device("ascend")
    else:
        config.disable_gpu()  # use cpu

    # create predictor
    predictor = paddle.inference.create_predictor(config)
    # random input
    input_data = np.random.random(size=[1, 1, 32, 32]).astype('float32')
    # set input
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape(input_data.shape)
    input_tensor.copy_from_cpu(input_data.copy())
    # Run
    predictor.run()
    # Set output
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])
    output_data = output_tensor.copy_to_cpu()
    print("Inference result is: ", np.argmax(output_data))


def main():
    args = parse_args()

    # set device
    paddle.set_device(args.device)

    # model
    model = LeNet5()

    # cost and optimizer
    cost = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters())

    # convert to ampo1 model
    if args.precision == "ampo1":
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level='O1')

    # convert to static model
    if args.to_static:
        build_strategy = paddle.static.BuildStrategy()
        mnist = paddle.jit.to_static(model, build_strategy=build_strategy)

    # data loader
    transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307, ), std=(0.3081, ))
    ])
    train_dataset = paddle.vision.datasets.MNIST(
        mode='train', transform=transform, download=True)
    test_dataset = paddle.vision.datasets.MNIST(
        mode='test', transform=transform, download=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # train and eval
    for epoch_id in range(EPOCH_NUM):
        if args.precision == "ampo1":
            train_func_ampo1(epoch_id, train_loader, model, cost, optimizer,
                             scaler)
        else:
            train_func_base(epoch_id, train_loader, model, cost, optimizer)
        test_func(epoch_id, test_loader, model, cost)

    # save inference model
    model.eval()
    if not args.to_static:
        mnist = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec([None, 1, 32, 32], 'float32', 'image')
            ])
        paddle.jit.save(model, 'build/lenet5')
    paddle.jit.save(model, 'build/lenet5')

    # inference
    infer_func('build/lenet5', device_type=args.device)


if __name__ == '__main__':
    main()
