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
# STRICT LIABILITY,OR TORT (INCLUDINGEargs NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import numpy as np
import paddle
from paddle import nn
from paddle import metric as M
from paddle.io import DataLoader
from paddle.nn import functional as F
from paddle.optimizer import Adam
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import Compose, Normalize
import unittest

BATCH_SIZE = 128
CLASS_DIM = 10
INIT_LR = 2e-4
EPOCHS = 4
LOG_GAP = 200

transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format="CHW")])
train_dataset = MNIST(mode="train", transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True
)


class DNN(nn.Layer):
    def __init__(self, n_classes=10):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.model(x)


class TestDy2Static(unittest.TestCase):
    def setUp(self):
        self.dy_model = DNN(n_classes=CLASS_DIM)
        self.static_model = paddle.jit.to_static(DNN(n_classes=CLASS_DIM))

    def get_result(self, model, step_num=1000):
        loss_arr, acc_arr = [], []
        model.train()
        opt = Adam(learning_rate=INIT_LR, parameters=model.parameters())
        i = 0
        for ep in range(EPOCHS):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                y_pred = model(x_data)
                acc = M.accuracy(y_pred, y_data)
                loss = F.cross_entropy(y_pred, y_data)
                acc_arr.append(acc.item())
                loss_arr.append(loss.item())
                opt.clear_grad()
                loss.backward()
                opt.step()
                i += 1
                if i >= step_num:
                    return loss_arr[200:], acc_arr[200:]

    def test_dy_and_static(self):
        dy_loss, dy_accr = self.get_result(self.dy_model, 1000)
        st_loss, st_accr = self.get_result(self.static_model, 1000)
        dy_out = [sum(dy_loss) / len(dy_loss), sum(dy_accr) / len(dy_accr)]
        st_out = [sum(st_loss) / len(st_loss), sum(st_accr) / len(st_accr)]
        np.testing.assert_allclose(
            dy_out,
            st_out,
            rtol=1e-2,
            atol=1e-2,
            err_msg="dy_model and to_statice_model out Not satisfied rtol : "
            + str(1e-2)
            + " atol : "
            + str(1e-2)
            + " dy mean_loss and mean_accr  : "
            + str(dy_out)
            + "st mean_loss and mean_accr  : "
            + str(st_out),
        )


if __name__ == "__main__":
    unittest.main()
