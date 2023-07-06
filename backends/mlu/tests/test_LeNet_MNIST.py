#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import time
import argparse
import datetime
import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
import paddle.inference as paddle_infer

EPOCH_NUM = 2
BATCH_SIZE = 4096


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "mlu"],
        default="mlu",
        help="Choose the device to run, it can be: cpu/gpu/mlu, default is mlu.",
    )
    parser.add_argument(
        "--ids", type=int, default=0, help="Choose the device id to run, default is 0."
    )
    parser.add_argument(
        "--amp",
        type=str,
        choices=["O0", "O1", "O2"],
        default="O1",
        help="Choose the amp level to run, default is O1.",
    )
    return parser.parse_args()


def test(epoch_id, test_loader, model, cost):
    model.eval()
    avg_acc = [[], []]
    for images, labels in test_loader():
        outputs = model(images)
        acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
        avg_acc[0].append(float(acc_top1))
        avg_acc[1].append(float(acc_top5))
    model.train()
    print(
        "Eval - Epoch ID: {}, Top1 accurary:: {:.5f}, Top5 accurary:: {:.5f}".format(
            epoch_id + 1, np.array(avg_acc[0]).mean(), np.array(avg_acc[1]).mean()
        )
    )


def infer(model_dir):
    # model file
    params_file = os.path.join(model_dir, "model.pdiparams")
    model_file = os.path.join(model_dir, "model.pdmodel")

    # create config
    config = paddle_infer.Config(model_file, params_file)
    config.enable_custom_device("mlu")

    # create predictor
    predictor = paddle_infer.create_predictor(config)

    # prepare input
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    # copy from cpu
    fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
    input_tensor.copy_from_cpu(fake_input)

    # run predictor
    predictor.run()

    # get output tensor
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    # get output data
    output_data = output_tensor.copy_to_cpu()
    print("Output data size is {}".format(output_data.size))
    print("Output data shape is {}".format(output_data.shape))


def main(args):
    model = paddle.vision.models.LeNet()
    cost = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters()
    )

    # convert to ampo1 model
    if args.amp == "O1":
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level="O1"
        )

    # data loader
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    train_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(mode="train", transform=transform, download=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=32,
        drop_last=True,
    )

    test_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(mode="test", transform=transform, download=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=32,
        drop_last=True,
    )

    # switch to train mode
    model.train()
    iter_max = len(train_loader)
    for epoch_id in range(EPOCH_NUM):
        batch_cost = AverageMeter("batch_cost", ":6.3f")
        reader_cost = AverageMeter("reader_cost", ":6.3f")

        # train
        epoch_start = time.time()
        tic = time.time()
        for iter_id, (images, labels) in enumerate(train_loader()):
            # reader_cost
            reader_cost.update(time.time() - tic)

            # forward
            if args.amp == "O1":
                # forward
                with paddle.amp.auto_cast(
                    custom_black_list={"flatten_contiguous_range", "greater_than"},
                    level="O1",
                ):
                    outputs = model(images)
                    loss = cost(outputs, labels)
                # backward and optimize
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
            else:
                # forward
                outputs = model(images)
                loss = cost(outputs, labels)
                # backward
                loss.backward()
                # optimize
                optimizer.minimize(loss)

            optimizer.clear_grad()

            # batch_cost and update tic
            batch_cost.update(time.time() - tic)
            tic = time.time()

            # logger for each step
            log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id)

        epoch_cost = time.time() - epoch_start
        avg_ips = iter_max * BATCH_SIZE / epoch_cost
        print(
            "Epoch ID: {}, Epoch time: {:.5f} s, reader_cost: {:.5f} s, batch_cost: {:.5f} s, avg ips: {:.5f} samples/s".format(
                epoch_id + 1,
                epoch_cost,
                reader_cost.sum,
                batch_cost.sum,
                avg_ips,
            )
        )

        # evaluate after each epoch
        test(epoch_id, test_loader, model, cost)

    # save inferece model
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=[None, 1, 28, 28], dtype="float32")],
    )
    paddle.jit.save(model, "output/model")

    # infernece and clear
    infer("output")
    shutil.rmtree("output")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name="", fmt="f", postfix="", need_avg=True):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_info(reader_cost, batch_cost, epoch_id, iter_max, iter_id):
    eta_sec = ((EPOCH_NUM - epoch_id) * iter_max - iter_id) * batch_cost.avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    print(
        "Epoch [{}/{}], Iter [{:0>2d}/{}], reader_cost: {:.5f} s, batch_cost: {:.5f} s, ips: {:.5f} samples/s, {}".format(
            epoch_id + 1,
            EPOCH_NUM,
            iter_id + 1,
            iter_max,
            reader_cost.avg,
            batch_cost.avg,
            BATCH_SIZE / batch_cost.avg,
            eta_msg,
        )
    )


if __name__ == "__main__":
    args = parse_args()
    paddle.set_device("{}:{}".format(args.device, str(args.ids)))
    main(args)
