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

import os
import tempfile
from unittest import TestCase

import json

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn, profiler
from paddle.io import DataLoader, Dataset


def filter_launch(event: dict):
    if not isinstance(event, dict):
        return False
    name = event.get("name")
    return isinstance(name, str) and "sdaalaunchkernel" in name.lower()


def filter_dnn_blas(event: dict):
    if not isinstance(event, dict):
        return False
    name = event.get("name")
    return isinstance(name, str) and ("dnn" in name.lower() or "blas" in name.lower())


def filter_kernel(event: dict):
    if not isinstance(event, dict):
        return False
    cat = event.get("cat")
    return isinstance(cat, str) and "kernel" in cat.lower()


class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, image, label=None):
        return self.fc(image)


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([100]).astype("float32")
        label = np.random.randint(0, 10 - 1, (1,)).astype("int64")
        return image, label

    def __len__(self):
        return self.num_samples


class TestProfiler(TestCase):
    def tearDown(self):
        self.temp_dir.cleanup()

    def check(self, trace):
        pass

    def make_profiler(self):
        return profiler.Profiler(on_trace_ready=lambda prof: None)

    def test_profiler(self):
        paddle.set_device("sdaa")
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()
        dataset = RandomDataset(10 * 4)
        simple_net = SimpleNet()
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
        opt = paddle.optimizer.Adam(
            learning_rate=1e-3,
            parameters=simple_net.parameters(),
            use_multi_tensor=True,
        )
        prof = self.make_profiler()
        prof.start()
        for i, (image, label) in enumerate(loader()):
            out = simple_net(image)
            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            opt.step()
            simple_net.clear_gradients()
            prof.step()
        prof.stop()

        path = os.path.join(self.temp_dir.name, "test_profiler.json")
        prof.export(path, format="json")
        with open(path, "r") as f:
            trace = json.load(f)
            events = trace["traceEvents"]
        self.check(events)
        paddle.enable_static()


class TestProfilerWithoutKernel(TestProfiler):
    def check(self, trace):
        launch_list = list(filter(filter_launch, trace))
        kernel_list = list(filter(filter_kernel, trace))
        dnn_blas_list = list(filter(filter_dnn_blas, trace))
        self.assertTrue(len(launch_list) == 0)
        self.assertTrue(len(kernel_list) == 0)
        self.assertTrue(len(dnn_blas_list) > 0)


class TestProfilerWithMultipleSchedule(TestProfiler):
    def make_profiler(self):
        return profiler.Profiler(
            on_trace_ready=lambda prof: None,
            targets=[
                profiler.ProfilerTarget.CPU,
                profiler.ProfilerTarget.CUSTOM_DEVICE,
            ],
            scheduler=profiler.make_scheduler(
                closed=1, ready=1, record=2, skip_first=1
            ),
        )
