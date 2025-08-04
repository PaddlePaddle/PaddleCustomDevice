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
import paddlenlp_ops


def round_up(value: int, k: int = 128) -> int:
    return (value + k - 1) // k * k


def pad_list(input, target_len, v):
    input_len = len(input)
    padding = target_len - input_len
    return input + [v] * padding


def prepare_block_metadata_ref(
    input_ids,
    rope_emb,
    block_tables,
    seq_lens_encoder,
    seq_lens_decoder,
    block_size,
    device_dtype,
):
    device = paddle.device.get_device()
    paddle.device.set_device("cpu")
    input_ids = input_ids.to("cpu")
    block_tables = block_tables.to("cpu")
    seq_lens_encoder = seq_lens_encoder.to("cpu")
    seq_lens_decoder = seq_lens_decoder.to("cpu")
    max_enc_len = paddle.max(seq_lens_encoder, axis=0).item()
    max_dec_len = paddle.max(seq_lens_decoder, axis=0).item()

    batch_step = 4
    block_step = 16

    is_prompt = None

    # prefill
    if max_enc_len > 0:
        is_prompt = True
        batch_ids = paddle.where(seq_lens_encoder > 0)[0].flatten().astype("int32")
        valid_batch = batch_ids.shape[0]

        input_tokens = paddle.index_select(input_ids, batch_ids)
        seq_lens = paddle.index_select(seq_lens_encoder, batch_ids).flatten()
        block_tables_seg = paddle.index_select(block_tables, batch_ids)

        total_batch = round_up(valid_batch, batch_step)

        max_buckets = (max_enc_len + block_size - 1) // block_size
        max_prompt_len = max_buckets * block_size

        src_padded = paddle.tensor.fill_constant(
            (total_batch, max_prompt_len), input_ids.dtype, 0, force_cpu=True
        )
        blk_padded = paddle.tensor.fill_constant(
            (total_batch, max_buckets), block_tables.dtype, -1, force_cpu=True
        )

        src_padded[:valid_batch, :max_prompt_len] = input_tokens[:, :max_prompt_len]
        blk_padded[:valid_batch, :max_buckets] = block_tables_seg[:, :max_buckets]
        block_indices = blk_padded.flatten().to(device)

        # rope_emb: [2, B=1, T=4096, 1, 128] --> [2, B=1, T=max_prompt_len, 1, 128]
        # Prefill:  [2, 1, T, 1, 128]
        rope_emb_seg = rope_emb[..., :max_prompt_len, :, :]
        rope_emb_seg = rope_emb_seg.to(device_dtype)

        block_offset = None
        block_groups = None
        block_list = None
        block_mapping = None
        attn_bias = None
        paddle.device.set_device(device)
    # decoding
    elif max_dec_len > 0:
        is_prompt = False
        batch_ids = paddle.where(seq_lens_decoder > 0)[0].flatten().astype("int32")
        valid_batch = batch_ids.shape[0]

        input_tokens = paddle.index_select(input_ids, batch_ids)[:, 0]
        seq_lens = paddle.index_select(seq_lens_decoder, batch_ids).flatten()
        block_tables_seg = paddle.index_select(block_tables, batch_ids)

        total_batch = round_up(valid_batch, batch_step)

        src_padded = paddle.tensor.fill_constant(
            (total_batch), input_ids.dtype, 0, force_cpu=True
        )
        seq_lens_padded = paddle.tensor.fill_constant(
            (total_batch), seq_lens_decoder.dtype, 0, force_cpu=True
        )
        block_indices_padded = paddle.tensor.fill_constant(
            (total_batch), block_tables.dtype, -1, force_cpu=True
        )
        block_offset_padded = paddle.tensor.fill_constant(
            (total_batch), block_tables.dtype, 0, force_cpu=True
        )

        src_padded[:valid_batch] = input_tokens[:]
        seq_lens_padded[:valid_batch] = seq_lens[:]

        last_block_pos = (seq_lens - 1) // block_size
        block_indices = paddle.index_sample(
            block_tables_seg, last_block_pos.unsqueeze(1)
        ).squeeze(1)
        block_offset = (seq_lens - 1) % block_size

        block_indices_padded[:valid_batch] = block_indices[:]
        block_offset_padded[:valid_batch] = block_offset[:]

        # rope_emb: [2, B=1, T=4096, 1, 128] --> [2, B=1, T=batch_size, 1, 128]
        # Decode : [2, 1, T, 1, 128] --> [2, T, 1, 1, 128]
        rope_emb_seg = (
            paddle.index_select(rope_emb, seq_lens_padded.to(device), 2)
            .squeeze(1)
            .unsqueeze(2)
        )
        rope_emb_seg = rope_emb_seg.to(device_dtype)

        block_list = []
        block_groups = []
        for group_index in range(valid_batch):
            block_list = block_list + (
                [
                    int(x)
                    for x in block_tables_seg[
                        group_index, : last_block_pos[group_index] + 1
                    ]
                ]
            )
            block_groups.extend(
                [group_index] * (last_block_pos[group_index].item() + 1)
            )

        block_bucket_size = round_up(len(block_list), block_step)
        padding_fn = lambda tensor, pad_value: pad_list(
            tensor, block_bucket_size, pad_value
        )

        block_list = padding_fn(block_list, -1)
        block_groups = padding_fn(block_groups, -1)
        block_groups_host = paddle.to_tensor(block_groups, dtype="float32", place="cpu")

        block_mapping = paddle.nn.functional.relu(block_groups_host)
        block_mapping = block_mapping.to("int32")
        block_mapping = paddle.nn.functional.one_hot(
            block_mapping, num_classes=total_batch
        )
        oob_values = block_groups_host.less_than(
            paddle.to_tensor(0, dtype="float32", place="cpu")
        )
        block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)

        block_usage = []
        for group_idx, length in enumerate(seq_lens):
            while length > 0:
                if length >= block_size:
                    block_usage.append(block_size)
                    length -= block_size
                else:
                    block_usage.append(length.item())
                    length = 0
        block_usage = padding_fn(block_usage, 1)
        block_usage = paddle.to_tensor(block_usage, dtype="int32")

        mask = paddle.arange(0, block_size, dtype="int32")
        mask = mask >= block_usage.unsqueeze(-1)
        attn_bias = paddle.zeros_like(mask, dtype="float32").masked_fill_(
            mask, float("-inf")
        )

        paddle.device.set_device(device)

        block_groups = paddle.to_tensor(block_groups).astype("int32")
        block_list = paddle.to_tensor(block_list).astype("int32")
        block_mapping = block_mapping.to(device_dtype).to(device)
        block_indices = block_indices_padded.to(device)
        block_offset = block_offset_padded.to(device)
        batch_ids = batch_ids.to(device).astype("int32")
        attn_bias = attn_bias.to(device_dtype).to(device)

    return (
        src_padded,
        rope_emb_seg,
        block_groups,
        block_list,
        block_indices,
        block_offset,
        block_mapping,
        attn_bias,
        batch_ids,
        is_prompt,
    )


def rebuild_padding_v2(
    tmp_out,
    batch_ids,
    total_batch,
    seq_lens_encoder,
    is_prompt=None,
):
    max_batch = seq_lens_encoder.shape[0]
    dim_emb = tmp_out.shape[-1]
    output_data = paddle.zeros((max_batch, dim_emb))

    if is_prompt is True:  # context
        tmp_out = tmp_out.reshape([total_batch, -1, dim_emb])
        j = 0
        for i in range(max_batch):
            if seq_lens_encoder[i].item() > 0:
                seq_len = seq_lens_encoder[i].item()
                output_data[i] = tmp_out[j, seq_len - 1]
                j = j + 1
    elif is_prompt is False:
        output_data = paddle.scatter(
            output_data, batch_ids, tmp_out[: batch_ids.shape[0], :]
        )

    return output_data


def rebuild_padding_v3(
    tmp_out,
    batch_ids,
    total_batch,
    seq_lens_encoder,
    is_prompt=None,
):
    dim_emb = tmp_out.shape[-1]
    output_data = paddle.zeros((batch_ids.shape[0], dim_emb))
    if is_prompt is True:  # context
        tmp_out = tmp_out.reshape([total_batch, -1, dim_emb])
        for i in range(batch_ids.shape[0]):
            seq_len = seq_lens_encoder[batch_ids[i]].item()
            output_data[i] = tmp_out[i, seq_len - 1]
    elif is_prompt is False:
        output_data = tmp_out[: batch_ids.shape[0], :]

    return output_data


def fused_flatpa_proj_ref(
    query,
    key_cache,
    value_cache,
    block_groups,
    block_list,
    block_mapping,
    block_bias,
    linear_weights,
    scaling_factor,
):
    batch_size = query.shape[0]
    q_heads = query.shape[2]
    head_size = query.shape[3]
    kv_heads = key_cache.shape[2]
    hidden_size = q_heads * head_size

    shape = tuple(query.shape)
    query = (
        paddle.matmul(block_mapping, (scaling_factor * query).view([shape[0], -1]))
        .view([-1, *shape[2:]])
        .unsqueeze(-2)
    )

    key = key_cache.index_select(block_list).transpose([0, 2, 1, 3])
    value = value_cache.index_select(block_list).transpose([0, 2, 1, 3])
    block_bias = block_bias.unsqueeze(1).unsqueeze(1)
    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose([0, 1, 2, 4, 3])
    else:
        key = key.transpose([0, 1, 3, 2])

    attn = paddle.matmul(query, key)
    # if 'fp32_softmax' in enabled_flags():
    #     attn = attn.float()
    attn = attn + block_bias

    block_max = attn.max(axis=-1, keepdim=True)
    adjustment_target_shape = block_max.shape
    attn = attn.subtract(block_max)
    attn = attn.exp()
    # attn = attn.to(value.dtype)
    block_sums = attn.sum(axis=-1, keepdim=True)
    attn = paddle.matmul(attn, value)
    block_max = block_max.squeeze()
    block_sums = block_sums.squeeze()

    # Calculate maximum of blocks that belong to the same sequences
    # and cast adjustments to native dtype
    orig_dtype = block_max.dtype
    if orig_dtype == paddle.float16:
        # fp16 index_reduce is not supported ATM
        block_max = block_max.to(paddle.float32)
    group_max = paddle.full(
        [batch_size + 1, *block_max.shape[1:]], float("-inf"), dtype=block_max.dtype
    )

    paddlenlp_ops.index_reduce_(group_max, block_groups, block_max, 0, "amax", True)
    group_max = group_max.index_select(block_groups, 0)

    block_adjustment = (block_max - group_max).exp()
    # block_adjustment = block_adjustment.to(value.dtype)
    sum_adjusted = block_sums.multiply(block_adjustment)

    # Sum block's sums that belongs to the same sequences
    shape = tuple(sum_adjusted.shape)
    group_sum_adjusted = paddle.matmul(
        block_mapping, sum_adjusted.view([shape[0], -1]), transpose_x=True
    ).view([-1, *shape[1:]])
    shape = tuple(group_sum_adjusted.shape)
    group_sum_adjusted = paddle.matmul(
        block_mapping, group_sum_adjusted.view([shape[0], -1])
    ).view([-1, *shape[1:]])

    sum_adjusted = sum_adjusted.view([*adjustment_target_shape])
    group_sum_adjusted = group_sum_adjusted.view([*adjustment_target_shape])
    block_adjustment = block_adjustment.view([*adjustment_target_shape])

    # For stability in case some of the sums have been zeroed out during block aggretation
    group_sum_adjusted = paddle.maximum(group_sum_adjusted, sum_adjusted)

    # Post processing for the attention scores
    rescale = block_adjustment.divide(group_sum_adjusted)
    attn = attn.multiply(rescale)

    shape = tuple(attn.shape)
    attn = paddle.matmul(
        block_mapping, attn.view([shape[0], -1]), transpose_x=True
    ).view([-1, *shape[1:]])

    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)

    return paddle.matmul(attn.view([batch_size, 1, hidden_size]), linear_weights)
