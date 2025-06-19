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

from typing import Any, Dict, List, Optional
import paddle
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


def rms_norm(x, weight, epsilon):
    output = core.eager._run_custom_op("rms_norm_gcu", x, weight, epsilon)[0]
    return output


def fused_add_rms_norm(x, residual, weight, epsilon):
    output, residual_update = core.eager._run_custom_op(
        "fused_add_rms_norm_op",
        x,
        residual,
        weight,
        epsilon,
    )
    return output, residual_update


def silu_and_mul(
    x,
):
    return paddle.incubate.nn.functional.swiglu(x)


def top_p_sampling(
    probs,
    top_p,
    version=1,
):
    if version == 1:
        sorted_probs, sorted_indices = paddle._C_ops.argsort(probs, -1, True, False)
        cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

        # Remove tokens with cumulative probs above the top_p, But keep at
        # least min_tokens_to_keep tokens
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep the first token
        sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")

        sorted_indices_to_remove = paddle.static.setitem(
            sorted_indices_to_remove,
            (slice(None), slice(1, None)),
            sorted_indices_to_remove[:, :-1].clone(),
        )
        sorted_indices_to_remove = paddle.static.setitem(
            sorted_indices_to_remove, (slice(None), 0), 0
        )

        # Scatter sorted tensors to original indexing
        sorted_indices = (
            sorted_indices
            + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
        )
        condition = paddle.scatter(
            sorted_indices_to_remove.flatten(),
            sorted_indices.flatten(),
            sorted_indices_to_remove.flatten(),
        )
        condition = paddle.cast(condition, "bool").reshape(probs.shape)
        probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
        next_tokens = paddle.multinomial(probs)
        next_scores = paddle.index_sample(probs, next_tokens)
        return next_scores, next_tokens
    elif version == 2:
        next_scores, next_tokens = core.eager._run_custom_op(
            "top_p_sampling_gcu",
            probs,
            top_p,
        )
        return next_scores, next_tokens
    else:
        raise (f"Not support top_p_sampling with version:{version}")


def topk_softmax(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    norm_topk_prob=False,
):
    topk_weights, topk_indices, token_expert_indices = core.eager._run_custom_op(
        "topk_softmax_gcu",
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        norm_topk_prob,
    )
    return topk_weights, topk_indices, token_expert_indices


def fused_moe_quant_kernel(
    C: paddle.Tensor,
    A: paddle.Tensor,
    B: paddle.Tensor,
    A_scale: Optional[paddle.Tensor],
    B_scale: paddle.Tensor,
    gs: int,
    B_zp: Optional[paddle.Tensor],
    topk_weights: paddle.Tensor,
    topk_ids: paddle.Tensor,
    sorted_token_ids: paddle.Tensor,
    experts_ids: paddle.Tensor,
    num_tokens_post_pad: paddle.Tensor,
    mul_routed_weight: bool,
    topk: int,
    block_size: int,
):
    c_out = core.eager._run_custom_op(
        "fused_moe_quant_kernel",
        C,
        A,
        B,
        A_scale,
        B_scale,
        B_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        gs,
        mul_routed_weight,
        topk,
        block_size,
    )

    return c_out


def fused_moe_kernel(
    C: paddle.Tensor,
    A: paddle.Tensor,
    B: paddle.Tensor,
    topk_weights: paddle.Tensor,
    topk_ids: paddle.Tensor,
    sorted_token_ids: paddle.Tensor,
    experts_ids: paddle.Tensor,
    num_tokens_post_pad: paddle.Tensor,
    mul_routed_weight: bool,
    topk: int,
    block_size: int,
    bias: Optional[paddle.Tensor],
):
    c_out = core.eager._run_custom_op(
        "fused_moe_kernel_gcu",
        C,
        A,
        B,
        bias,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        mul_routed_weight,
        topk,
        block_size,
    )

    return c_out


def moe_align_block_size(
    sorted_token_ids: paddle.Tensor,
    experts_ids: paddle.Tensor,
    num_tokens_post_pad: paddle.Tensor,
    topk_ids: paddle.Tensor,
    num_experts: int,
    block_size: int,
):
    (
        sorted_token_ids_out,
        experts_ids_out,
        num_tokens_post_pad_out,
    ) = core.eager._run_custom_op(
        "moe_align_block_size_gcu",
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        topk_ids,
        num_experts,
        block_size,
    )

    return sorted_token_ids_out, experts_ids_out, num_tokens_post_pad_out


def invoke_fused_moe_kernel(
    A: paddle.Tensor,  # input
    B: paddle.Tensor,  # weight
    C: paddle.Tensor,  # output
    A_scale: Optional[paddle.Tensor],  # w8a8 input scale
    B_scale: Optional[paddle.Tensor],
    B_zp: Optional[paddle.Tensor],
    topk_weights: paddle.Tensor,
    topk_ids: paddle.Tensor,
    sorted_token_ids: paddle.Tensor,
    expert_ids: paddle.Tensor,
    num_tokens_post_padded: paddle.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    use_int4_w4a16: bool,
    block_shape: Optional[List[int]] = None,
) -> None:
    assert topk_weights.strides[1] == 1
    assert sorted_token_ids.strides[0] == 1

    if use_int4_w4a16:
        assert B_scale is not None
        assert block_shape and block_shape[0] == 0
        assert B_zp is None or B_zp.ndim == 3

    block_size = config["BLOCK_SIZE_M"]

    if use_int4_w4a16:
        A_scale = None
        group_size = block_shape[1]

        fused_moe_quant_kernel(
            C,
            A,
            B,
            A_scale,
            B_scale,
            group_size,
            B_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            block_size,
        )
    else:
        topk_weights = topk_weights.astype("float32")  # WA for grouped_topk
        fused_moe_kernel(
            C,
            A,
            B,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            block_size,
            None,
        )


def awq_gemm(
    input: paddle.Tensor,
    qweight: paddle.Tensor,
    scales: paddle.Tensor,
    qzeros=None,
    bias=None,
    group_size=128,
):
    if qzeros is None:
        qzeros = paddle.zeros_like(scales)

    linear_output = (
        core.eager._run_custom_op(
            "awq_gemm_gcu", input, qweight, scales, qzeros, bias, group_size
        )
    )[0]

    return linear_output


def weight_only_quant(
    input: paddle.Tensor,
    qweight: paddle.Tensor,
    scales: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
    group_size=-1,
):
    linear_output = (
        core.eager._run_custom_op(
            "weight_only_quant_gcu", input, qweight, bias, scales, group_size
        )
    )[0]

    return linear_output
