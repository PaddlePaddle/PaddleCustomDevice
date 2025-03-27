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
import unittest
import paddle
import paddlenlp_ops

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def min_length_logits_process(
    logits, cur_len, min_len, eos_token_id, bs, length, end_length
):
    for bi in range(bs):
        if cur_len[bi] < 0:
            continue
        if cur_len[bi] < min_len[bi]:
            for i in range(end_length):
                logits[bi, eos_token_id[i]] = -1e10


def update_repeat_times(pre_ids, cur_len, repeat_times, bs, length_id):
    for bi in range(bs):
        if cur_len[bi] < 0:
            continue
        for i in range(length_id):
            id = pre_ids[bi][i]
            if id < 0:
                break
            repeat_times[bi][id] += 1


def update_value_by_repeat_times(
    repeat_times, penalty_scores, frequency_score, presence_score, logits, bs, length
):
    for bi in range(bs):
        alpha = penalty_scores[bi]
        beta = frequency_score[bi]
        gamma = presence_score[bi]
        for i in range(length):
            times = repeat_times[bi][i]
            if times == 0:
                continue
            logit_now = logits[bi][i]
            if logit_now < 0:
                logit_now = logit_now * alpha
            else:
                logit_now = logit_now / alpha
            logits[bi][i] = logit_now - times * beta - gamma


def ref_get_token_penalty_multi_scores(
    pre_ids,
    logits,
    penalty_scores,
    frequency_scores,
    presence_scores,
    cur_len,
    min_len,
    eos_token_id,
    use_cpu=True,
):
    shape = logits.shape
    repeat_times = paddle.full(shape, 0, dtype="int32").to(pre_ids.place)
    bs = shape[0]
    length = shape[1]
    length_id = pre_ids.shape[1]
    logits_out = logits.clone()
    end_length = eos_token_id.shape[0]

    min_length_logits_process(
        logits_out, cur_len, min_len, eos_token_id, bs, length, end_length
    )
    update_repeat_times(pre_ids, cur_len, repeat_times, bs, length_id)
    update_value_by_repeat_times(
        repeat_times,
        penalty_scores,
        frequency_scores,
        presence_scores,
        logits_out,
        bs,
        length,
    )

    return logits_out


def clone(*tensors):
    cloned = []
    for tensor in tensors:
        cloned_tensor = tensor.clone().cpu()
        cloned.append(cloned_tensor)

    if len(cloned) == 1:
        return cloned[0]
    else:
        return tuple(cloned)


class get_token_penalty_multi_scores_test(unittest.TestCase):
    def is_token_in_targets(self, tokens, target_tensors):
        for target in target_tensors:
            if paddle.all(paddle.equal(tokens, target)):
                return True
        return False

    # cur_lens < min_lens
    # eos at zero-slot
    def test_case1(self):
        pre_ids = paddle.to_tensor(
            [[3, 1, 6, 1, -1, -1], [1, 4, 5, 5, -1, -1]], dtype="int64"
        )
        logits = paddle.to_tensor(
            [
                [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
                [0.8, 0.7, 0.6, 0.5, -0.4, 0.3, 0.2, 0.1],
            ],
            dtype="float32",
        )
        penalty = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
        frequency = paddle.to_tensor([[0.1], [0.1]], dtype="float32")
        presence = paddle.to_tensor([[0.2], [0.2]], dtype="float32")
        cur_lens = paddle.to_tensor([[4], [4]], dtype="int64")
        min_lens = paddle.to_tensor([[32], [32]], dtype="int64")
        eos_token_id = paddle.to_tensor([[0], [0]], dtype="int64")

        (
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        ) = clone(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out = paddlenlp_ops.get_token_penalty_multi_scores(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out_ref = ref_get_token_penalty_multi_scores(
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        )
        logits_out_ref = logits_out_ref.to(logits_out.place)

        self.assertTrue(paddle.all(paddle.equal(logits_out, logits_out_ref)))

    # cur_lens > min_lens
    # eos at zero-slot
    def test_case2(self):
        pre_ids = paddle.to_tensor(
            [[3, 1, 6, 1, -1, -1], [1, 4, 5, 5, -1, -1]], dtype="int64"
        )
        logits = paddle.to_tensor(
            [
                [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
                [0.8, 0.7, 0.6, 0.5, -0.4, 0.3, 0.2, 0.1],
            ],
            dtype="float32",
        )
        penalty = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
        frequency = paddle.to_tensor([[0.1], [0.1]], dtype="float32")
        presence = paddle.to_tensor([[0.2], [0.2]], dtype="float32")
        cur_lens = paddle.to_tensor([[4], [4]], dtype="int64")
        min_lens = paddle.to_tensor([[1], [1]], dtype="int64")
        eos_token_id = paddle.to_tensor([[0], [0]], dtype="int64")

        (
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        ) = clone(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out = paddlenlp_ops.get_token_penalty_multi_scores(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out_ref = ref_get_token_penalty_multi_scores(
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        )
        logits_out_ref = logits_out_ref.to(logits_out.place)

        self.assertTrue(paddle.all(paddle.equal(logits_out, logits_out_ref)))

    # cur_lens < min_lens
    # eos at non-zero slot
    def test_case3(self):
        pre_ids = paddle.to_tensor(
            [[3, 1, 6, 1, -1, -1], [1, 4, 5, 5, -1, -1]], dtype="int64"
        )
        logits = paddle.to_tensor(
            [
                [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
                [0.8, 0.7, 0.6, 0.5, -0.4, 0.3, 0.2, 0.1],
            ],
            dtype="float32",
        )
        penalty = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
        frequency = paddle.to_tensor([[0.1], [0.1]], dtype="float32")
        presence = paddle.to_tensor([[0.2], [0.2]], dtype="float32")
        cur_lens = paddle.to_tensor([[4], [4]], dtype="int64")
        min_lens = paddle.to_tensor([[32], [32]], dtype="int64")
        eos_token_id = paddle.to_tensor([[2], [2]], dtype="int64")

        (
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        ) = clone(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out = paddlenlp_ops.get_token_penalty_multi_scores(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out_ref = ref_get_token_penalty_multi_scores(
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        )
        logits_out_ref = logits_out_ref.to(logits_out.place)

        self.assertTrue(paddle.all(paddle.equal(logits_out, logits_out_ref)))

    # cur_lens > min_lens
    # eos at non-zero slot
    def test_case4(self):
        pre_ids = paddle.to_tensor(
            [[3, 1, 6, 1, -1, -1], [1, 4, 5, 5, -1, -1]], dtype="int64"
        )
        logits = paddle.to_tensor(
            [
                [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
                [0.8, 0.7, 0.6, 0.5, -0.4, 0.3, 0.2, 0.1],
            ],
            dtype="float32",
        )
        penalty = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
        frequency = paddle.to_tensor([[0.1], [0.1]], dtype="float32")
        presence = paddle.to_tensor([[0.2], [0.2]], dtype="float32")
        cur_lens = paddle.to_tensor([[4], [4]], dtype="int64")
        min_lens = paddle.to_tensor([[1], [1]], dtype="int64")
        eos_token_id = paddle.to_tensor([[6], [6]], dtype="int64")

        (
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        ) = clone(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out = paddlenlp_ops.get_token_penalty_multi_scores(
            pre_ids,
            logits,
            penalty,
            frequency,
            presence,
            cur_lens,
            min_lens,
            eos_token_id,
        )

        logits_out_ref = ref_get_token_penalty_multi_scores(
            pre_ids_ref,
            logits_ref,
            penalty_ref,
            frequency_ref,
            presence_ref,
            cur_lens_ref,
            min_lens_ref,
            eos_token_id_ref,
        )
        logits_out_ref = logits_out_ref.to(logits_out.place)

        self.assertTrue(paddle.all(paddle.equal(logits_out, logits_out_ref)))


if __name__ == "__main__":
    unittest.main()
