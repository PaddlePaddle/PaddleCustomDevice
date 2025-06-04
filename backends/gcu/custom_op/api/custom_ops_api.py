#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List, Optional
from paddle.base import core


def mem_efficient_attention(
    q,
    k,
    v,
    softmax_scale,
    attn_bias=None,
    dropout=0.0,
    mask_mode=1,
    seqlens: Optional[List[int]] = None,
):
    seq_lens = seqlens if seqlens is not None else [0]
    attn_output = core.eager._run_custom_op(
        "mem_efficient_attention_gcu",
        q,
        k,
        v,
        attn_bias,
        dropout,
        softmax_scale,
        mask_mode,
        seq_lens,
        True,  # casual
    )[0]
    return attn_output


def flash_attn_var_len(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    leftpad_k=None,
    block_table=None,
    alibi_slopes=None,
    p_dropout=0.0,
    softmax_scale=None,
    zero_tensors=False,
    is_causal=True,
):
    assert (
        cu_seqlens_k is not None or seqused_k is not None
    ), "cu_seqlens_k or seqused_k must be provided"
    assert (
        cu_seqlens_k is None or seqused_k is None
    ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert (
        block_table is None or seqused_k is not None
    ), "seqused_k must be provided if block_table is provided"

    attn_output = core.eager._run_custom_op(
        "flash_attn_var_len_gcu",
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap = 0.0 means deactivated
        False,  # return_softmax
    )[0]
    return attn_output


def reshape_and_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype="auto",
    k_scale=1.0,
    k_zero=0.0,
    v_scale=1.0,
    v_zero=0.0,
):
    key_cache_out, value_cache_out = core.eager._run_custom_op(
        "reshape_and_cache_gcu",
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        k_zero,
        v_scale,
        v_zero,
    )
    return key_cache_out, value_cache_out


def paged_attention(
    q,
    k_cache,
    v_cache,
    num_kv_heads,
    scale,
    block_tables,
    seq_lens,
    block_size,
    max_seq_len,
    kv_cache_dtype="auto",
    k_scale=1.0,
    k_zero=0.0,
    v_scale=1.0,
    v_zero=0.0,
    alibi_slopes=None,
    out_scales=None,
):
    attn_output = core.eager._run_custom_op(
        "paged_attention_gcu",
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        alibi_slopes,
        out_scales,
        num_kv_heads,
        scale,
        block_size,
        max_seq_len,
        kv_cache_dtype,
        k_scale,
        k_zero,
        v_scale,
        v_zero,
    )[0]
    return attn_output
