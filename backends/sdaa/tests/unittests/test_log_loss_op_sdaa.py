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
SEED = 1234


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


class TestLogLossOp(OpTest):
    def setUp(self):
        self.op_type = "log_loss"
        samples_num = 100
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)
        x = np.random.random((samples_num, 1)).astype("float32")
        predicted = sigmoid_array(x)
        labels = np.random.randint(0, 2, (samples_num, 1)).astype("float32")
        epsilon = 1e-7
        self.inputs = {
            "Predicted": predicted,
            "Labels": labels,
        }

        self.attrs = {"epsilon": epsilon}
        loss = -labels * np.log(predicted + epsilon) - (1 - labels) * np.log(
            1 - predicted + epsilon
        )
        self.outputs = {"Loss": loss}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["Predicted"], "Loss", check_dygraph=True
        )


if __name__ == "__main__":
    unittest.main()
