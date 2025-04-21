# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddlenlp_ops import step_paddle
import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")


def main():
    # Define input shapes and parameters
    bsz = 2  # Batch size
    block_size = 128
    encoder_decoder_block_num = 4
    max_block_num = 16
    max_seq_len = 4096
    # Initialize inputs
    stop_flags = paddle.to_tensor([[False], [False]], dtype="bool")
    seq_lens_this_time = paddle.full(shape=[bsz, 1], fill_value=8, dtype="int32")
    step_seq_lens_encoder = paddle.full(shape=[bsz, 1], fill_value=8, dtype="int32")
    seq_lens_encoder = paddle.full(shape=[bsz, 1], fill_value=8, dtype="int32")
    seq_lens_decoder = paddle.full(shape=[bsz, 1], fill_value=0, dtype="int32")
    block_tables = paddle.full(shape=[bsz, max_block_num], fill_value=-1, dtype="int32")
    encoder_block_lens = paddle.full(shape=[bsz], fill_value=2, dtype="int32")
    is_block_step = paddle.full(shape=[bsz], fill_value=False, dtype="bool")
    step_block_list = paddle.full(shape=[bsz], fill_value=-1, dtype="int32")
    step_lens = paddle.full(shape=[1], fill_value=0, dtype="int32")
    recover_block_list = paddle.full(shape=[bsz], fill_value=-1, dtype="int32")
    recover_lens = paddle.full(shape=[1], fill_value=0, dtype="int32")
    need_block_list = paddle.full(shape=[bsz], fill_value=-1, dtype="int32")
    need_block_len = paddle.full(shape=[1], fill_value=0, dtype="int32")
    used_list_len = paddle.full(shape=[bsz], fill_value=0, dtype="int32")

    free_list = list(range(max_block_num - 1, 11, -1))
    free_list_len = len(free_list)
    free_list = paddle.to_tensor(free_list, dtype="int32")
    free_list_len = paddle.full(shape=[1], fill_value=free_list_len, dtype="int32")

    input_ids = paddle.full(shape=[bsz, max_seq_len], fill_value=0, dtype="int64")
    pre_ids = paddle.full(shape=[bsz, max_seq_len], fill_value=-1, dtype="int64")
    step_idx = paddle.full(shape=[bsz, 1], fill_value=0, dtype="int64")
    next_tokens = paddle.full(shape=[bsz, 1], fill_value=101, dtype="int64")
    first_token_ids = paddle.full(shape=[bsz, 1], fill_value=100, dtype="int64")

    # Call the custom operator
    step_paddle(
        stop_flags,
        seq_lens_this_time,
        step_seq_lens_encoder,
        seq_lens_encoder,
        seq_lens_decoder,
        block_tables,
        encoder_block_lens,
        is_block_step,
        step_block_list,
        step_lens,
        recover_block_list,
        recover_lens,
        need_block_list,
        need_block_len,
        used_list_len,
        free_list,
        free_list_len,
        input_ids,
        pre_ids,
        step_idx,
        next_tokens,
        first_token_ids,
        block_size=block_size,
        encoder_decoder_block_num=encoder_decoder_block_num,
    )

    # Verify outputs
    print("stop_flags:", stop_flags.numpy())
    print("block_tables:", block_tables.numpy())
    print("free_list:", free_list.numpy())
    print("free_list_len:", free_list_len.numpy())
    assert free_list_len.numpy()[0] == bsz, "Not dispatch blocks"


if __name__ == "__main__":
    main()
