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

import paddle
import numpy as np
import unittest
from op_test import OpTest

paddle.enable_static()


def nll_loss_1d(logs, targets, weight=None, reduction="mean", ignore_index=-100):
    input_shape = logs.shape
    N = input_shape[0]
    C = input_shape[1]
    out = np.zeros_like(targets).astype(np.float64)
    total_weight = 0
    for i in range(N):
        cur_target = targets[i]
        if cur_target == ignore_index:
            out[i] = 0
            continue
        cur_weight = weight[cur_target] if weight is not None else 1
        total_weight += cur_weight
        out[i] = -logs[i][cur_target] * cur_weight
    if reduction == "sum":
        return np.sum(out), np.array([total_weight]).astype("float64")
    elif reduction == "mean":
        return out.sum() / total_weight, np.array([total_weight]).astype("float64")
    elif reduction == "none":
        return out


def nll_loss_2d(logs, targets, weight=None, reduction="mean", ignore_index=-100):
    input_shape = logs.shape
    N = input_shape[0]
    H = input_shape[2]
    W = input_shape[3]
    out = np.zeros_like(targets).astype(np.float64)
    total_weight = 0
    for i in range(N):
        for h in range(H):
            for w in range(W):
                cur_target = targets[i][h][w]
                if cur_target == ignore_index:
                    out[i][h][w] = 0
                    continue
                cur_weight = weight[cur_target] if weight is not None else 1
                total_weight += cur_weight
                out[i][h][w] = -logs[i][cur_target][h][w] * cur_weight
    if reduction == "sum":
        return np.sum(out), np.array([total_weight]).astype("float64")
    elif reduction == "mean":
        return out.sum() / total_weight, np.array([total_weight]).astype("float64")
    elif reduction == "none":
        return out


class TestNLLLossOp1DWithReduce(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_test_case()
        self.init_test_dtype()
        self.op_type = "nll_loss"
        self.python_api = paddle.nn.functional.nll_loss
        self.python_out_sig = ["Out"]
        self.with_weight = False
        np.random.seed(200)
        input_np = np.random.uniform(0.1, 0.8, self.input_shape).astype(self.dtype)
        np.random.seed(200)
        label_np = np.random.randint(0, self.input_shape[1], self.label_shape).astype(
            "int64"
        )
        output_np, total_weight_np = nll_loss_1d(input_np, label_np)
        self.inputs = {"X": input_np, "Label": label_np}
        if self.with_weight:
            np.random.seed(200)
            weight_np = np.random.uniform(0.1, 0.8, self.input_shape[1]).astype(
                self.dtype
            )
            output_np, total_weight_np = nll_loss_1d(
                input_np, label_np, weight=weight_np
            )
            self.inputs["Weight"] = weight_np

        self.outputs = {"Out": output_np, "Total_weight": total_weight_np[0]}
        self.attrs = {"reduction": "mean", "ignore_index": -100}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.with_weight = True
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.input_shape = [10, 10]
        self.label_shape = [10]

    def init_test_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestNLLLossOp1DWithReduceFP16(TestNLLLossOp1DWithReduce):
    def init_test_dtype(self):
        self.dtype = np.float16


class TestNLLLossOp1DNoReduce(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_test_case()
        self.init_test_dtype()
        self.op_type = "nll_loss"
        self.python_api = paddle.nn.functional.nll_loss
        self.python_out_sig = ["Out"]
        self.with_weight = False
        np.random.seed(200)
        input_np = np.random.uniform(0.1, 0.8, self.input_shape).astype(self.dtype)
        np.random.seed(200)
        label_np = np.random.randint(0, self.input_shape[1], self.label_shape).astype(
            "int64"
        )
        output_np = nll_loss_1d(input_np, label_np, reduction="none")
        total_weight_np = np.array([0]).astype("float64")
        self.inputs = {"X": input_np, "Label": label_np}
        if self.with_weight:
            np.random.seed(200)
            weight_np = np.random.uniform(0.1, 0.8, self.input_shape[1]).astype(
                self.dtype
            )
            output_np, total_weight_np = nll_loss_1d(
                input_np, label_np, weight=weight_np, reduction="none"
            )
            self.inputs["Weight"] = weight_np

        self.outputs = {"Out": output_np, "Total_weight": total_weight_np[0]}
        self.attrs = {"reduction": "none", "ignore_index": -100}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.with_weight = True
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.input_shape = [10, 10]
        self.label_shape = [10]

    def init_test_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestNLLLossOp1DNoReduceFP16(TestNLLLossOp1DNoReduce):
    def init_test_dtype(self):
        self.dtype = np.float16


class TestNLLLossOp2DWithReduce(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_test_case()
        self.init_test_dtype()
        self.op_type = "nll_loss"
        self.python_api = paddle.nn.functional.nll_loss
        self.python_out_sig = ["Out"]
        self.with_weight = False
        np.random.seed(200)
        input_np = np.random.uniform(0.1, 0.8, self.input_shape).astype(self.dtype)
        np.random.seed(200)
        label_np = np.random.randint(0, self.input_shape[1], self.label_shape).astype(
            "int64"
        )
        output_np, total_weight_np = nll_loss_2d(input_np, label_np)
        self.inputs = {"X": input_np, "Label": label_np}
        if self.with_weight:
            np.random.seed(200)
            weight_np = np.random.uniform(0.1, 0.8, self.input_shape[1]).astype(
                self.dtype
            )
            output_np, total_weight_np = nll_loss_2d(
                input_np, label_np, weight=weight_np
            )
            self.inputs["Weight"] = weight_np

        self.outputs = {"Out": output_np, "Total_weight": total_weight_np[0]}
        self.attrs = {"reduction": "mean", "ignore_index": -100}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.with_weight = True
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.input_shape = [2, 3, 5, 5]
        self.label_shape = [2, 5, 5]

    def init_test_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestNLLLossOp2DWithReduceFP16(TestNLLLossOp2DWithReduce):
    def init_test_dtype(self):
        self.dtype = np.float16


class TestNLLLossOp2DNoReduce(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_test_case()
        self.init_test_case()
        self.op_type = "nll_loss"
        self.python_api = paddle.nn.functional.nll_loss
        self.python_out_sig = ["Out"]
        self.with_weight = False
        np.random.seed(200)
        input_np = np.random.uniform(0.1, 0.8, self.input_shape).astype(self.dtype)
        np.random.seed(200)
        label_np = np.random.randint(0, self.input_shape[1], self.label_shape).astype(
            "int64"
        )
        output_np = nll_loss_2d(input_np, label_np, reduction="none")
        total_weight_np = np.array([0]).astype("float32")
        self.inputs = {"X": input_np, "Label": label_np}
        if self.with_weight:
            np.random.seed(200)
            weight_np = np.random.uniform(0.1, 0.8, self.input_shape[1]).astype(
                self.dtype
            )
            output_np, total_weight_np = nll_loss_2d(
                input_np, label_np, weight=weight_np, reduction="none"
            )
            self.inputs["Weight"] = weight_np

        self.outputs = {"Out": output_np, "Total_weight": total_weight_np[0]}
        self.attrs = {"reduction": "none", "ignore_index": -100}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.with_weight = True
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.input_shape = [5, 3, 5, 5]
        self.label_shape = [5, 5, 5]

    def init_test_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestNLLLossOp2DNoReduceFP16(TestNLLLossOp2DNoReduce):
    def init_test_dtype(self):
        self.dtype = np.float16


if __name__ == "__main__":
    unittest.main()
