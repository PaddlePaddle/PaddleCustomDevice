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

import numpy as np
import paddle
from paddlenlp_ops import set_preids_token_penalty_multi_scores

paddle.seed(2023)


def tensors_to_cpu(*tensors):
    return [tensor.cpu() for tensor in tensors]


def tensors_to_device(device, *tensors):
    return [tensor.to(device) for tensor in tensors]


def min_length_logits_process(
    logits, cur_len, min_len, eos_token_id, bs, length, end_length
):
    mask = cur_len < min_len
    neg_inf = paddle.to_tensor([-1e10], "float32", place=logits.place)

    for i in range(bs):
        if mask[i][0]:
            for j in range(end_length):
                eos = eos_token_id[0][j]
                logits[i][eos] = neg_inf


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
    repeat_times,
    penalty_scores,
    frequency_score,
    presence_score,
    temperatures,
    logits,
    bs,
    length,
):
    for bi in range(bs):
        alpha = penalty_scores[bi]
        beta = frequency_score[bi]
        gamma = presence_score[bi]
        temperature = temperatures[bi]
        for i in range(length):
            times = repeat_times[bi][i]
            logit_now = logits[bi][i]
            if times != 0:
                if logit_now < 0:
                    logit_now = logit_now * alpha
                else:
                    logit_now = logit_now / alpha
                logit_now = logit_now - times * beta - gamma
            logit_now = logit_now / temperature
            logits[bi][i] = logit_now


def logits_mark_bad_tokens(logits, bad_tokens, bs, length):
    neg_inf = paddle.to_tensor([-1e10], "float32", place=logits.place)

    for i in range(bs):
        for j in range(length):
            idx = bad_tokens[j]
            logits[i][idx] = neg_inf


def ref_calc_scores(
    pre_ids,
    logits,
    penalty_scores,
    frequency_scores,
    presence_scores,
    temperatures,
    bad_tokens,
    cur_len,
    min_len,
    eos_token_id,
):
    logits_out = logits.clone()

    device = pre_ids.place
    (
        pre_ids,
        logits,
        logits_out,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
    ) = tensors_to_cpu(
        pre_ids,
        logits,
        logits_out,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
    )

    shape = logits.shape
    repeat_times = paddle.full(shape, 0, dtype="int32").to(pre_ids.place)
    bs = shape[0]
    length = shape[1]
    length_id = pre_ids.shape[1]
    end_length = eos_token_id.shape[1]
    bad_token_length = bad_tokens.shape[0]

    min_length_logits_process(
        logits_out, cur_len, min_len, eos_token_id, bs, length, end_length
    )
    update_repeat_times(pre_ids, cur_len, repeat_times, bs, length_id)
    update_value_by_repeat_times(
        repeat_times,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        logits_out,
        bs,
        length,
    )
    logits_mark_bad_tokens(logits_out, bad_tokens, bs, bad_token_length)

    (
        pre_ids,
        logits,
        logits_out,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        cur_len,
        min_len,
        repeat_times,
        eos_token_id,
    ) = tensors_to_device(
        device,
        pre_ids,
        logits,
        logits_out,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        cur_len,
        min_len,
        repeat_times,
        eos_token_id,
    )

    return logits_out


def ref_fill_inputs(pre_ids, input_ids, len_enc, len_dec, step_idx, stop_flags):
    dim0, dim1 = pre_ids.shape

    zero = paddle.zeros_like(len_enc, dtype=len_enc.dtype)
    valid_src_idx = paddle.where(len_enc > 0, len_enc - 1, zero)
    input_ids = paddle.gather(input_ids, valid_src_idx, axis=1)

    pre_ids.flatten_()
    step_idx.flatten_()
    len_enc.flatten_()
    len_dec.flatten_()
    stop_flags.flatten_()
    input_ids.flatten_()

    valid_step_idx = paddle.where(step_idx >= 0, step_idx, 0)
    condition = (step_idx >= 0) & (~stop_flags)
    dst_idx = paddle.arange(0, dim0) * dim1 + valid_step_idx
    dst_idx = dst_idx[condition]
    src_idx = paddle.nonzero(condition)

    selected_elements = paddle.gather(input_ids, src_idx)
    pre_ids.scatter_(dst_idx, selected_elements)

    # restore the shape of output tensor
    paddle.reshape_(pre_ids, [dim0, dim1])


class SetPreidsTokenPenaltyMultiScores(unittest.TestCase):
    # both pre_ids and input_ids smaller than 64
    def test_block_pp_fill_inputs_1(self):
        pre_ids = paddle.to_tensor(
            [[1, 9, 3, 4, 5, 6, 7, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1]],
            "int64",
        )
        input_ids = paddle.to_tensor(
            [
                [1, 9, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1],
                [1, 9, 7, 6, 5, 4, -1, -1, -1, -1, -1, -1, -1],
            ],
            "int64",
        )
        seq_lens_encoder = paddle.to_tensor([[1], [1]], "int32")
        seq_lens_decoder = paddle.to_tensor([[1], [1]], "int32")
        step_idx = paddle.to_tensor([[1], [1]], "int64")
        stop_flags = paddle.to_tensor([[0], [1]], "bool")

        logits = paddle.to_tensor(
            [
                [0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.1, 0.1, 0.1, 0.1],
            ],
            "float32",
        )
        penalty_scores = paddle.to_tensor([[1.0], [1.0]], "float32")
        frequency_scores = paddle.to_tensor([[0.1], [0.1]], "float32")
        presence_scores = paddle.to_tensor([[0.0], [0.0]], "float32")
        temperatures = paddle.to_tensor([[0.5], [0.25]], "float32")
        bad_tokens = paddle.to_tensor([0, 1], "int64")
        cur_len = paddle.to_tensor([[7], [6]], "int64")
        min_len = paddle.to_tensor([[1], [8]], "int64")
        eos_token_id = paddle.to_tensor([[2, 9]], "int64")

        pre_ids_ref = pre_ids.clone()
        input_ids_ref = input_ids.clone()
        len_enc_ref = seq_lens_encoder.clone()
        len_dec_ref = seq_lens_decoder.clone()
        step_idx_ref = step_idx.clone()
        stop_flags_ref = stop_flags.clone()
        ref_fill_inputs(
            pre_ids_ref,
            input_ids_ref,
            len_enc_ref,
            len_dec_ref,
            step_idx_ref,
            stop_flags_ref,
        )

        set_preids_token_penalty_multi_scores(
            pre_ids,
            input_ids,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            stop_flags,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        pre_ids_np = pre_ids.numpy()
        pre_ids_ref_np = pre_ids_ref.numpy()

        np.testing.assert_array_equal(
            pre_ids_np,
            pre_ids_ref_np,
            err_msg="pre_ids arrays are different between runs",
        )

        print("Test passed: The two runs produced identical results.")

    # both pre_ids smaller than 64 and input_ids larger than 64
    def test_block_pp_fill_inputs_2(self):
        pos = 80
        pre_ids = paddle.to_tensor(
            [[1, 9, 3, 4, 5, 6, 7, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1]],
            "int64",
        )
        base = paddle.arange(0, 100)
        input_ids = paddle.stack([base] * 2, axis=0).to("int64")
        seq_lens_encoder = paddle.to_tensor([[pos], [1]], "int32")
        seq_lens_decoder = paddle.to_tensor([[1], [1]], "int32")
        step_idx = paddle.to_tensor([[1], [1]], "int64")
        stop_flags = paddle.to_tensor([[0], [1]], "bool")

        logits = paddle.to_tensor(
            [
                [0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.1, 0.1, 0.1, 0.1],
            ],
            "float32",
        )
        penalty_scores = paddle.to_tensor([[1.0], [1.0]], "float32")
        frequency_scores = paddle.to_tensor([[0.1], [0.1]], "float32")
        presence_scores = paddle.to_tensor([[0.0], [0.0]], "float32")
        temperatures = paddle.to_tensor([[0.5], [0.25]], "float32")
        bad_tokens = paddle.to_tensor([0, 1], "int64")
        cur_len = paddle.to_tensor([[7], [6]], "int64")
        min_len = paddle.to_tensor([[1], [8]], "int64")
        eos_token_id = paddle.to_tensor([[2, 9]], "int64")

        pre_ids_ref = pre_ids.clone()
        input_ids_ref = input_ids.clone()
        len_enc_ref = seq_lens_encoder.clone()
        len_dec_ref = seq_lens_decoder.clone()
        step_idx_ref = step_idx.clone()
        stop_flags_ref = stop_flags.clone()
        ref_fill_inputs(
            pre_ids_ref,
            input_ids_ref,
            len_enc_ref,
            len_dec_ref,
            step_idx_ref,
            stop_flags_ref,
        )

        set_preids_token_penalty_multi_scores(
            pre_ids,
            input_ids,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            stop_flags,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        pre_ids_np = pre_ids.numpy()
        pre_ids_ref_np = pre_ids_ref.numpy()

        np.testing.assert_array_equal(
            pre_ids_np,
            pre_ids_ref_np,
            err_msg="pre_ids arrays are different between runs",
        )
        print("Test passed: The two runs produced identical results.")

    # both pre_ids larger than 64 and input_ids smaller than 64
    def test_block_pp_fill_inputs_3(self):
        pos = 80
        pre_ids = paddle.zeros(shape=[2, 100], dtype="int64")
        input_ids = paddle.to_tensor(
            [
                [1, 9, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1],
                [1, 9, 7, 6, 5, 4, -1, -1, -1, -1, -1, -1, -1],
            ],
            "int64",
        )
        seq_lens_encoder = paddle.to_tensor([[1], [1]], "int32")
        seq_lens_decoder = paddle.to_tensor([[1], [1]], "int32")
        step_idx = paddle.to_tensor([[1], [pos]], "int64")
        stop_flags = paddle.to_tensor([[1], [0]], "bool")

        logits = paddle.to_tensor(
            [
                [0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.1, 0.1, 0.1, 0.1],
            ],
            "float32",
        )
        penalty_scores = paddle.to_tensor([[1.0], [1.0]], "float32")
        frequency_scores = paddle.to_tensor([[0.1], [0.1]], "float32")
        presence_scores = paddle.to_tensor([[0.0], [0.0]], "float32")
        temperatures = paddle.to_tensor([[0.5], [0.25]], "float32")
        bad_tokens = paddle.to_tensor([0, 1], "int64")
        cur_len = paddle.to_tensor([[7], [6]], "int64")
        min_len = paddle.to_tensor([[1], [8]], "int64")
        eos_token_id = paddle.to_tensor([[2, 9]], "int64")

        pre_ids_ref = pre_ids.clone()
        input_ids_ref = input_ids.clone()
        len_enc_ref = seq_lens_encoder.clone()
        len_dec_ref = seq_lens_decoder.clone()
        step_idx_ref = step_idx.clone()
        stop_flags_ref = stop_flags.clone()
        ref_fill_inputs(
            pre_ids_ref,
            input_ids_ref,
            len_enc_ref,
            len_dec_ref,
            step_idx_ref,
            stop_flags_ref,
        )

        set_preids_token_penalty_multi_scores(
            pre_ids,
            input_ids,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            stop_flags,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        pre_ids_np = pre_ids.numpy()
        pre_ids_ref_np = pre_ids_ref.numpy()
        # pre_ids_ref_np[1][pos] = 1

        np.testing.assert_array_equal(
            pre_ids_np,
            pre_ids_ref_np,
            err_msg="pre_ids arrays are different between runs",
        )
        print("Test passed: The two runs produced identical results.")

    # eos offset between 0 and 64
    def test_block_pp_set_eos_1(self):
        pre_ids = paddle.to_tensor(
            [[1, 9, 3, 4, 5, 6, 7, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1]],
            "int64",
        )
        input_ids = paddle.to_tensor(
            [
                [1, 9, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1],
                [1, 9, 7, 6, 5, 4, -1, -1, -1, -1, -1, -1, -1],
            ],
            "int64",
        )
        seq_lens_encoder = paddle.to_tensor([[1], [1]], "int32")
        seq_lens_decoder = paddle.to_tensor([[1], [1]], "int32")
        step_idx = paddle.to_tensor([[1], [1]], "int64")
        stop_flags = paddle.to_tensor([[0], [1]], "bool")

        logits = paddle.to_tensor(
            [
                [0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.1, 0.1, 0.1, 0.1],
            ],
            "float32",
        )
        logits_ref = logits.clone()
        penalty_scores = paddle.to_tensor([[1.0], [1.0]], "float32")
        frequency_scores = paddle.to_tensor([[0.1], [0.1]], "float32")
        presence_scores = paddle.to_tensor([[0.0], [0.0]], "float32")
        temperatures = paddle.to_tensor([[0.5], [0.25]], "float32")
        bad_tokens = paddle.to_tensor([0, 1], "int64")
        cur_len = paddle.to_tensor([[7], [6]], "int64")
        min_len = paddle.to_tensor([[1], [8]], "int64")
        eos_token_id = paddle.to_tensor([[2, 9]], "int64")

        pre_ids_ref = pre_ids.clone()
        input_ids_ref = input_ids.clone()
        len_enc_ref = seq_lens_encoder.clone()
        len_dec_ref = seq_lens_decoder.clone()
        step_idx_ref = step_idx.clone()
        stop_flags_ref = stop_flags.clone()
        ref_fill_inputs(
            pre_ids_ref,
            input_ids_ref,
            len_enc_ref,
            len_dec_ref,
            step_idx_ref,
            stop_flags_ref,
        )

        logits_ref = ref_calc_scores(
            pre_ids_ref,
            logits_ref,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        set_preids_token_penalty_multi_scores(
            pre_ids,
            input_ids,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            stop_flags,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        logits_np = logits.numpy()
        logits_ref_np = logits_ref.numpy()

        np.testing.assert_array_equal(
            logits_np, logits_ref_np, err_msg="logits arrays are different between runs"
        )
        print("Test passed: The two runs produced identical results.")

    # eos offset > 64
    def test_block_pp_set_eos_2(self):
        pre_ids = paddle.to_tensor(
            [[1, 9, 3, 4, 5, 6, 7, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1]],
            "int64",
        )
        input_ids = paddle.to_tensor(
            [
                [1, 9, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1],
                [1, 9, 7, 6, 5, 4, -1, -1, -1, -1, -1, -1, -1],
            ],
            "int64",
        )
        seq_lens_encoder = paddle.to_tensor([[1], [1]], "int32")
        seq_lens_decoder = paddle.to_tensor([[1], [1]], "int32")
        step_idx = paddle.to_tensor([[1], [1]], "int64")
        stop_flags = paddle.to_tensor([[0], [1]], "bool")

        logits = paddle.zeros(shape=[2, 100], dtype="float32")
        penalty_scores = paddle.to_tensor([[1.0], [1.0]], "float32")
        frequency_scores = paddle.to_tensor([[0.1], [0.1]], "float32")
        presence_scores = paddle.to_tensor([[0.0], [0.0]], "float32")
        temperatures = paddle.to_tensor([[0.5], [0.25]], "float32")
        bad_tokens = paddle.to_tensor([0, 1], "int64")
        cur_len = paddle.to_tensor([[6], [7]], "int64")
        min_len = paddle.to_tensor([[8], [1]], "int64")
        eos_token_id = paddle.to_tensor([[12, 69]], "int64")

        pre_ids_ref = pre_ids.clone()
        input_ids_ref = input_ids.clone()
        len_enc_ref = seq_lens_encoder.clone()
        len_dec_ref = seq_lens_decoder.clone()
        step_idx_ref = step_idx.clone()
        stop_flags_ref = stop_flags.clone()
        ref_fill_inputs(
            pre_ids_ref,
            input_ids_ref,
            len_enc_ref,
            len_dec_ref,
            step_idx_ref,
            stop_flags_ref,
        )

        logits_ref = logits.clone()
        logits_ref = ref_calc_scores(
            pre_ids_ref,
            logits_ref,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        set_preids_token_penalty_multi_scores(
            pre_ids,
            input_ids,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            stop_flags,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        logits_np = logits.numpy()
        logits_ref_np = logits_ref.numpy()

        np.testing.assert_array_equal(
            logits_np, logits_ref_np, err_msg="logits arrays are different between runs"
        )
        print("Test passed: The two runs produced identical results.")


if __name__ == "__main__":
    unittest.main()
