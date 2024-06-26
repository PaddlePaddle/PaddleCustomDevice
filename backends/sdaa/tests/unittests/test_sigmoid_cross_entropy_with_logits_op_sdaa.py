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
from scipy.special import logit
from scipy.special import expit

paddle.enable_static()
SEED = 2022


def loss_wrapper(logit, label, pos_weight=None, normalize=False, ignore_index=-100):
    out = paddle._C_ops.sigmoid_cross_entropy_with_logits(
        logit, label, pos_weight, normalize, ignore_index
    )
    return out


class TestSigmoidCrossEntropyWithLogitsOp1(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = 64
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(self.dtype)
            ),
            "Label": np.random.randint(0, 2, (batch_size, num_classes)).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(paddle.CustomPlace("sdaa", 0), ["X"], "Out")
        else:
            self.check_grad_with_place(
                paddle.CustomPlace("sdaa", 0),
                ["X"],
                "Out",
                numeric_place=paddle.CPUPlace(),
            )

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32


class TestSigmoidCrossEntropyWithLogitsOp3(TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = 64
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(self.dtype)
            ),
            "Label": np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}


class TestSigmoidCrossEntropyWithLogitsOp5(TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes])).astype(
                    self.dtype
                )
            ),
            "Label": np.random.uniform(0, 1, tuple(batch_size + [num_classes])).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}


class TestSigmoidCrossEntropyWithLogitsOp6(TestSigmoidCrossEntropyWithLogitsOp1):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.set_sdaa()
        self.init_dtype()

        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            "X": logit(
                np.random.uniform(0, 1, tuple(batch_size + [num_classes])).astype(
                    self.dtype
                )
            ),
            "Label": np.random.randint(0, 2, tuple(batch_size + [num_classes])).astype(
                self.dtype
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs["X"])
        term1 = self.inputs["Label"] * np.log(sigmoid_X)
        term2 = (1 - self.inputs["Label"]) * np.log(1 - sigmoid_X)
        self.outputs = {"Out": -term1 - term2}


class TestSigmoidCrossEntropyWithLogitsOp7(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabalistic label"""

    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        batch_size = 64
        num_classes = 20

        x = logit(np.random.uniform(0, 1, (batch_size, num_classes)).astype("float32"))
        label = np.random.uniform(0, 1, (batch_size, num_classes)).astype("float32")
        pos_weight = np.ones((batch_size, num_classes)).astype("float32")
        self.inputs = {
            "X": x,
            "Label": label,
            "pos_weight": pos_weight,
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        term1 = np.maximum(self.inputs["X"], 0)
        term2 = self.inputs["X"] * self.inputs["Label"]
        term3 = np.log(1 + np.exp(-1 * np.abs(self.inputs["X"]))) * pos_weight
        self.outputs = {"Out": term1 - term2 + term3}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CustomPlace("sdaa", 0), ["X"], "Out", numeric_place=paddle.CPUPlace()
        )


if __name__ == "__main__":
    unittest.main()
