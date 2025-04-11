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


def get_padding_offset_v2(
    input_ids, cum_offset, token_num, seq_lens, draft_tokens=None, seq_lens_encoder=None
):
    bsz, max_seq_len = input_ids.shape
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens)
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens)
    x_remove_padding = paddle.zeros(shape=(token_num), dtype=input_ids.dtype)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    current_index = 0
    for i in range(bsz):
        seq_len_now = seq_lens[i].item()
        cum_offset = cum_offsets[i].item()
        # x_remove_padding = paddle.concat((x_remove_padding, input_ids[i, :seq_len_now]))
        x_remove_padding[current_index : current_index + seq_len_now] = input_ids[
            i, :seq_len_now
        ]
        current_index += seq_len_now
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1].item()
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return (
        x_remove_padding,
        cum_offsets[:-1],
        padding_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
    )


def rebuild_padding_v2(
    tmp_out,
    cum_offsets,
    seq_lens_decoder,
    seq_len_encoder,
    output_padding_offset=None,
    max_len=-1,
):
    # tmp_out, // [token_num, dim_embed]
    # cum_offsets, // [bsz, 1]
    bs = seq_len_encoder.shape[0]
    dim_emb = tmp_out.shape[1]
    output_data = paddle.zeros((bs, dim_emb)).flatten()
    seq_len = max_len
    tmp_out = tmp_out.flatten()
    for i in range(bs * dim_emb):
        bi = i // dim_emb
        bias_idx = i % dim_emb
        seq_id = 0
        # just encoder or stop, get last token; just decoder, get first token.
        if seq_lens_decoder[bi] == 0:
            if seq_len_encoder[bi] != 0:
                seq_id = seq_len_encoder[bi] - 1
            else:
                continue
        ori_token_idx = bi * seq_len - cum_offsets[bi] + seq_id
        src_offset = ori_token_idx * dim_emb + bias_idx
        output_data[i] = tmp_out[src_offset]
    return output_data.reshape([bs, dim_emb])


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
