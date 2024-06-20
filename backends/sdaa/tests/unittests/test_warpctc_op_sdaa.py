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

import sys
import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle.base import Program, program_guard

CUDA_BLOCK_SIZE = 32


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class CTCForward:
    def __init__(
        self,
        softmax,
        softmax_lod,
        labels,
        labels_lod,
        num_classes,
        batch_size,
        blank,
        norm_by_times,
    ):
        self.softmax = softmax
        self.softmax_lod = softmax_lod
        self.labels = labels
        self.labels_lod = labels_lod
        self.blank = blank
        self.norm_by_times = norm_by_times

        self.level = 0
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.loss = np.zeros([self.batch_size, 1], dtype=softmax.dtype)
        self.gradient = np.zeros(self.softmax.shape, dtype=softmax.dtype)

        # float64
        self.EXP_MAX = sys.float_info.max
        self.EXP_MIN = sys.float_info.min
        self.LOG_ZERO = np.log(self.EXP_MIN)
        self.LOG_INFINITY = np.log(self.EXP_MAX)

    def safe_exp(self, x):
        if x <= self.LOG_ZERO:
            return 0.0
        if x >= self.LOG_INFINITY:
            return self.EXP_MAX
        return np.exp(x)

    def safe_log(self, x):
        if x <= self.EXP_MIN:
            return self.LOG_ZERO
        return np.log(x)

    # x = lna and y = lnb are in log scale, ln(a / b) = lna - lnb
    def log_div(self, x, y):
        res = x - y
        if res <= self.LOG_ZERO:
            return self.LOG_ZERO
        if res >= self.LOG_INFINITY:
            return self.LOG_INFINITY
        return res

    # x = lna and y = lnb are in log scale, ln(a * b) = lna + lnb
    def log_mul(self, x, y):
        res = x + y
        if res <= self.LOG_ZERO:
            return self.LOG_ZERO
        if res >= self.LOG_INFINITY:
            return self.LOG_INFINITY
        return res

    # x = lna and y = lnb are in log scale,
    # ln(a + b) = lna + ln(1 + exp(lnb - lna)), where b > a
    def log_add(self, x, y):
        if x < y:
            t = y
            y = x
            x = t
        return x + self.safe_log(1 + self.safe_exp(y - x))

    def segment_range(self, time, total_times, total_segments):
        start = max(0, total_segments - (2 * (total_times - time)))
        end = min(total_segments, 2 * (time + 1))
        return start, end

    def forward_a_sequence(self, softmax_a_sequence, labels_a_sequence):
        total_times = softmax_a_sequence.shape[0]
        total_segments = labels_a_sequence.shape[0] * 2 + 1

        required_times = labels_a_sequence.shape[0]
        old_label = -1
        for i in range(labels_a_sequence.shape[0]):
            # two contingous labels with the same value
            if labels_a_sequence[i, 0] == old_label:
                required_times = required_times + 1
            old_label = labels_a_sequence[i, 0]

        if total_times < required_times:
            return 0

        # calculate the forward and backward variables,
        # reference Chapter 7.3 of "Alex Grave, Supervised Sequence
        # Labelling with Recurrent Neural Networks"
        log_acts = np.zeros(
            [total_times, self.num_classes], dtype=softmax_a_sequence.dtype
        )
        for i in range(total_times):
            for j in range(self.num_classes):
                log_acts[i, j] = self.safe_log(softmax_a_sequence[i, j])

        # calculate the forward variables
        forward_vars = np.zeros(
            [total_times, total_segments], dtype=softmax_a_sequence.dtype
        )
        for i in range(total_times):
            for j in range(total_segments):
                forward_vars[i, j] = self.LOG_ZERO

        for i in range(total_times):
            # dp initialization at t0
            if i == 0:
                forward_vars[i, 0] = log_acts[0, self.blank]
                if total_segments > 1:
                    forward_vars[i, 1] = log_acts[0, labels_a_sequence[i, 0]]
                continue

            # dp from t1
            start, end = self.segment_range(i, total_times, total_segments)
            for k in range(end - start):
                j = k + start
                if j & 1 == 1:
                    label_idx = j // 2
                    label_val = labels_a_sequence[label_idx, 0]
                    fv = self.log_add(
                        forward_vars[i - 1, j], forward_vars[i - 1, j - 1]
                    )
                    if j > 1 and label_val != labels_a_sequence[label_idx - 1, 0]:
                        fv = self.log_add(fv, forward_vars[i - 1, j - 2])
                    fv = self.log_mul(fv, log_acts[i, label_val])
                else:
                    fv = forward_vars[i - 1, j]
                    if j > 0:
                        fv = self.log_add(fv, forward_vars[i - 1, j - 1])
                    fv = self.log_mul(fv, log_acts[i, self.blank])
                forward_vars[i, j] = fv

        # sum the last two value as log_prob
        log_prob = forward_vars[total_times - 1, total_segments - 1]
        if total_segments > 1:
            log_prob = self.log_add(
                log_prob, forward_vars[total_times - 1, total_segments - 2]
            )

        return -log_prob

    def forward(self):
        softmax_offset = 0
        labels_offset = 0
        for i in range(self.batch_size):
            if self.labels.shape[1] == 1:
                softmax_start_i = softmax_offset
                softmax_end_i = softmax_offset + self.softmax_lod[self.level][i]
                labels_start_i = labels_offset
                labels_end_i = labels_offset + self.labels_lod[self.level][i]

                softmax_a_sequence = self.softmax[softmax_start_i:softmax_end_i, :]
                labels_a_sequence = self.labels[labels_start_i:labels_end_i, :]
                self.loss[i] = self.forward_a_sequence(
                    softmax_a_sequence, labels_a_sequence
                )
                softmax_offset += self.softmax_lod[self.level][i]
                labels_offset += self.labels_lod[self.level][i]
            else:
                softmax_a_sequence = self.softmax[: self.softmax_lod[i], i, :]
                labels_a_sequence = self.labels[: self.labels_lod[i], :]
                self.loss[i] = self.forward_a_sequence(
                    softmax_a_sequence, labels_a_sequence
                )

        return self.loss


def warpctc_wrapper(
    Logits,
    Label,
    LogitsLength=None,
    LabelLength=None,
    blank=0,
    norm_by_times=False,
):
    return paddle._C_ops.warpctc(
        Logits, Label, LogitsLength, LabelLength, blank, norm_by_times
    )


@skip_check_grad_ci(
    reason="warpctc_grad check numeric_grads has problems, use api test to check grads compute."
)
class TestWarpCTCOp(OpTest):
    def config(self):
        self.batch_size = 4
        self.num_classes = 8
        self.logits_lod = [[4, 1, 5, 5]]
        self.labels_lod = [[3, 1, 4, 2]]
        self.logits_length = np.array([4, 1, 5, 5], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 2], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

    def setUp(self):
        self.set_sdaa()
        self.op_type = "warpctc"
        self.python_api = warpctc_wrapper
        self.python_out_sig = ["Loss"]
        self.place = paddle.CustomPlace("sdaa", 0)
        self.config()

        logits = np.random.uniform(
            0.1, 1.0, [sum(self.logits_length), self.num_classes]
        ).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        # labels should not be blank
        labels = np.random.randint(
            0, self.num_classes - 1, [sum(self.labels_length), 1], dtype="int32"
        )

        ctc = CTCForward(
            softmax,
            self.logits_lod,
            labels,
            self.labels_lod,
            self.num_classes,
            self.batch_size,
            self.blank,
            self.norm_by_times,
        )
        loss = ctc.forward()

        max_sequence_length = 0
        for i in range(self.batch_size):
            max_sequence_length = max(max_sequence_length, self.logits_length[i])
        # reshape logits to T*N*S
        new_logits = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype,
        )

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.logits_length[batch_id]):
                for j in range(self.num_classes):
                    new_logits[i, batch_id, j] = logits[cur + i, j]
            cur = cur + self.logits_length[batch_id]

        self.gradient = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype,
        )

        self.inputs = {
            "Logits": new_logits,
            "Label": (labels, self.labels_lod),
            "LogitsLength": self.logits_length,
            "LabelLength": self.labels_length,
        }
        self.outputs = {"Loss": loss}
        self.attrs = {
            "blank": self.blank,
            "norm_by_times": self.norm_by_times,
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestWarpCTCOpCase1(TestWarpCTCOp):
    def config(self):
        self.batch_size = 4
        self.num_classes = CUDA_BLOCK_SIZE + 2
        self.logits_lod = [[4, 1, 3, 3]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 3, 3], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False


@skip_check_grad_ci(
    reason="warpctc_grad check numeric_grads has problems, use api test to check grads compute."
)
class TestWarpCTCOpWithPadding(OpTest):
    def config(self):
        self.batch_size = 4
        self.num_classes = 8
        self.logits_lod = [[4, 1, 3, 3]]
        self.labels_lod = [[3, 1, 4, 4]]
        self.logits_length = np.array([4, 1, 3, 3], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 4], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False

    def setUp(self):
        self.set_sdaa()
        self.op_type = "warpctc"
        self.python_api = warpctc_wrapper
        self.python_out_sig = ["Loss"]
        self.place = paddle.CustomPlace("sdaa", 0)
        self.config()

        logits = np.random.uniform(
            0.1, 1.0, [sum(self.logits_length), self.num_classes]
        ).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        # labels should not be blank
        labels = np.random.randint(
            0, self.num_classes - 1, [sum(self.labels_length), 1], dtype="int32"
        )

        ctc = CTCForward(
            softmax,
            self.logits_lod,
            labels,
            self.labels_lod,
            self.num_classes,
            self.batch_size,
            self.blank,
            self.norm_by_times,
        )
        loss = ctc.forward()

        max_sequence_length = 0
        for i in range(self.batch_size):
            max_sequence_length = max(max_sequence_length, self.logits_length[i])
        # reshape logits to T*N*S
        new_logits = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype,
        )

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.logits_length[batch_id]):
                for j in range(self.num_classes):
                    new_logits[i, batch_id, j] = logits[cur + i, j]
            cur = cur + self.logits_length[batch_id]

        # reshape labels to N*S
        max_target_seq_length = 0
        for i in range(self.batch_size):
            max_target_seq_length = max(max_target_seq_length, self.labels_length[i])
        new_labels = np.zeros([self.batch_size, max_target_seq_length], dtype="int32")

        cur = 0
        for batch_id in range(self.batch_size):
            for i in range(self.labels_length[batch_id]):
                new_labels[batch_id, i] = labels[cur + i]
            cur = cur + self.labels_length[batch_id]

        self.gradient = np.zeros(
            [max_sequence_length, self.batch_size, self.num_classes],
            dtype=logits.dtype,
        )

        self.inputs = {
            "Logits": new_logits,
            "Label": new_labels,
            "LogitsLength": self.logits_length,
            "LabelLength": self.labels_length,
        }
        self.outputs = {"Loss": loss}
        self.attrs = {
            "blank": self.blank,
            "norm_by_times": self.norm_by_times,
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestWarpCTCOpWithPaddingCase1(TestWarpCTCOpWithPadding):
    def config(self):
        self.batch_size = 4
        self.num_classes = CUDA_BLOCK_SIZE + 2
        self.logits_lod = [[4, 1, 5, 5]]
        self.labels_lod = [[3, 1, 4, 2]]
        self.logits_length = np.array([4, 1, 5, 5], dtype=np.int64)
        self.labels_length = np.array([3, 1, 4, 2], dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False


class TestWarpCTCOpModelShapeCase(TestWarpCTCOpWithPadding):
    def config(self):
        self.batch_size = 64
        self.num_classes = 6625
        self.logits_lod = [
            [
                4,
                1,
                3,
                80,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        ]
        self.labels_lod = [
            [
                3,
                1,
                4,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                25,
                1,
                1,
            ]
        ]
        self.logits_length = np.array(
            [
                4,
                1,
                3,
                80,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=np.int64,
        )
        self.labels_length = np.array(
            [
                3,
                1,
                4,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                13,
                1,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                25,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                25,
                1,
                1,
            ],
            dtype=np.int64,
        )
        self.blank = self.num_classes - 1
        self.norm_by_times = False


class TestWarpCTCGradAPICase(unittest.TestCase):

    paddle.set_device("sdaa")
    loss_func = paddle.nn.CTCLoss(blank=0, reduction="none")

    np_logits = np.random.uniform(0.1, 1.0, [80, 64, 6625]).astype("float32")
    np_labels = np.random.randint(1, 6625, [64, 25], dtype="int32")
    np_logits_length = np.array(
        [
            4,
            1,
            3,
            80,
            1,
            1,
            13,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            4,
            25,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            13,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            13,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            13,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            4,
            25,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=np.int64,
    )
    np_labels_length = np.array(
        [
            3,
            1,
            4,
            4,
            25,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            13,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            4,
            25,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            13,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            13,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            4,
            25,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            4,
            25,
            1,
            1,
        ],
        dtype=np.int64,
    )
    predicts = paddle.to_tensor(np_logits, stop_gradient=False)
    labels = paddle.to_tensor(np_labels)
    preds_lengths = paddle.to_tensor(np_logits_length)
    label_lengths = paddle.to_tensor(np_labels_length)

    loss = loss_func(predicts, labels, preds_lengths, label_lengths)
    loss.backward()

    paddle.set_device("cpu")
    loss_cpu_func = paddle.nn.CTCLoss(blank=0, reduction="none")
    predicts_cpu = predicts._to("cpu")
    labels_cpu = labels._to("cpu")
    preds_lengths_cpu = preds_lengths._to("cpu")
    label_lengths_cpu = label_lengths._to("cpu")

    losscpu = loss_cpu_func(
        predicts_cpu, labels_cpu, preds_lengths_cpu, label_lengths_cpu
    )
    losscpu.backward()

    np.testing.assert_allclose(loss, losscpu, rtol=1e-05, atol=1)
    np.testing.assert_allclose(predicts.grad, predicts_cpu.grad, rtol=1e-05, atol=1)


class TestWarpCTCOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        paddle.set_device("sdaa")
        with program_guard(Program(), Program()):
            logits = paddle.static.data(
                name="logits", shape=[5, 16, 6], dtype="float32"
            )
            logits_length = paddle.static.data(
                name="logits_length", shape=[None], dtype="int64"
            )
            label = paddle.static.data(name="label", shape=[16, 3], dtype="int32")
            label_length = paddle.static.data(
                name="labels_length", shape=[None], dtype="int64"
            )

            def test_logits_Variable():
                logits_data = np.random.rand(5, 16, 6).astype(logits.dtype)
                paddle.nn.functional.ctc_loss(
                    log_probs=logits_data,
                    labels=label,
                    input_lengths=logits_length,
                    label_lengths=label_length,
                    reduction="none",
                )

            self.assertRaises(TypeError, test_logits_Variable)

            def test_label_Variable():
                label_data = np.random.randint(0, 5, [5, 1]).astype("int32")
                paddle.nn.functional.ctc_loss(
                    log_probs=logits,
                    labels=label_data,
                    input_lengths=logits_length,
                    label_lengths=label_length,
                    reduction="none",
                )

            self.assertRaises(TypeError, test_label_Variable)

            def test_logits_len_Variable():
                logits_length_data = np.array([5] * 16).astype("int64")
                paddle.nn.functional.ctc_loss(
                    log_probs=logits,
                    labels=label,
                    input_lengths=logits_length_data,
                    label_lengths=label_length,
                    reduction="none",
                )

            self.assertRaises(TypeError, test_logits_len_Variable)

            def test_label_len_Variable():
                label_length_data = np.array([3] * 16).astype("int64")
                paddle.nn.functional.ctc_loss(
                    log_probs=logits,
                    labels=label,
                    input_lengths=logits_length,
                    label_length=label_length_data,
                    reduction="none",
                )

            self.assertRaises(TypeError, test_label_len_Variable)

    def test_dygraph_errors(self):
        def test_dygraph_with_lod():

            paddle.set_device("sdaa")
            logits = np.random.uniform(0.1, 1.0, [20, 15]).astype("float32")
            # labels should not be blank
            labels = np.random.randint(0, 15 - 1, [15, 1], dtype="int32")
            softmax = paddle.to_tensor(logits)
            labels = paddle.to_tensor(labels)

            paddle.nn.functional.ctc_loss(
                log_probs=softmax,
                labels=labels,
                input_lengths=None,
                label_lengths=None,
                reduction="none",
            )

        paddle.disable_static()
        self.assertRaises(ValueError, test_dygraph_with_lod)
        paddle.enable_static()

    def test_no_length_errors(self):
        def test_no_logits_length():
            paddle.set_device("sdaa")
            logits = np.random.uniform(0.1, 1.0, [5, 4, 34]).astype("float32")
            labels = np.random.randint(0, 33, [4, 4], dtype="int32")
            logits_length = np.array([4, 1, 5, 5], dtype=np.int64)
            label_length = np.array([3, 1, 4, 2], dtype=np.int64)
            softmax = paddle.to_tensor(logits)
            labels = paddle.to_tensor(labels)
            input_lengths = paddle.to_tensor(logits_length)
            label_lengths = paddle.to_tensor(label_length)

            warpctc_wrapper(softmax, labels, None, label_lengths)

        def test_no_label_length():
            paddle.set_device("sdaa")
            logits = np.random.uniform(0.1, 1.0, [5, 4, 34]).astype("float32")
            labels = np.random.randint(0, 33, [4, 4], dtype="int32")
            logits_length = np.array([4, 1, 5, 5], dtype=np.int64)
            label_length = np.array([3, 1, 4, 2], dtype=np.int64)
            softmax = paddle.to_tensor(logits)
            labels = paddle.to_tensor(labels)
            input_lengths = paddle.to_tensor(logits_length)
            label_lengths = paddle.to_tensor(label_length)

            warpctc_wrapper(softmax, labels, input_lengths, None)

        def test_no_length():
            paddle.set_device("sdaa")
            logits = np.random.uniform(0.1, 1.0, [5, 4, 34]).astype("float32")
            labels = np.random.randint(0, 33, [4, 4], dtype="int32")
            logits_length = np.array([4, 1, 5, 5], dtype=np.int64)
            label_length = np.array([3, 1, 4, 2], dtype=np.int64)
            softmax = paddle.to_tensor(logits)
            labels = paddle.to_tensor(labels)
            input_lengths = paddle.to_tensor(logits_length)
            label_lengths = paddle.to_tensor(label_length)

            warpctc_wrapper(softmax, labels, None, None)

        paddle.disable_static()
        self.assertRaises(NotImplementedError, test_no_logits_length)
        self.assertRaises(NotImplementedError, test_no_label_length)
        self.assertRaises(NotImplementedError, test_no_length)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
