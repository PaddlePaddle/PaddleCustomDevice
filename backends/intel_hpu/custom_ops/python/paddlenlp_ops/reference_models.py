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
import paddle.distributed as dist
import paddlenlp_ops
import os

# import logging

measure_dict = {}
rank = dist.get_rank()
world_size = dist.get_world_size()
if world_size == 1:
    model_measurement_file = "./model_measurement.txt"
else:
    model_measurement_file = f"./model_measurement_{rank}.txt"


def init_measure_dict():
    global measure_dict
    if os.path.exists(model_measurement_file):
        with open(model_measurement_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split("\t")
                measure_dict[key] = float(value)


def save_measure_dict():
    with open(model_measurement_file, "w") as f:
        for key, value in measure_dict.items():
            f.write(f"{key}\t{value}\n")


def measure_matrix(amax_in, key, experts_min=0, experts_max=0):
    global measure_dict

    if isinstance(amax_in, paddle.Tensor):
        if amax_in.shape == [1] or len(amax_in.shape) == 0:
            amax_in = float(amax_in.item())
            prev_val = measure_dict.get(key, float("-inf"))
            new_val = max(prev_val, amax_in)
            measure_dict[key] = new_val
        elif len(amax_in.shape) == 1 and amax_in.shape[0] > 1:
            results = []
            assert (
                amax_in.shape[0] == experts_max - experts_min + 1
            ), f"Assertion failed: Expect amax_in.shape[0](={amax_in.shape[0]}) = experts_max(={experts_max}) -  experts_min(={experts_min}) + 1"
            for i in range(experts_min, experts_max + 1):
                subkey = key.format(i)
                val = float(amax_in[i - experts_min].item())
                prev_val = measure_dict.get(subkey, float("-inf"))
                new_val = max(prev_val, val)
                measure_dict[subkey] = new_val
                results.append(new_val)
        else:
            raise ValueError("Unsupported tensor shape for measure_matrix")
    else:
        prev_val = measure_dict.get(key, float("-inf"))
        new_val = max(prev_val, float(amax_in))
        measure_dict[key] = new_val


def fused_qkv_rope_ref(
    src,
    qkv_weights,
    qkv_biases,
    rotary_embs,
    head_dim,
    num_head,
    total_batch,
    transpose,
    use_neox_style,
    measurement_mode=False,
    qkv_act_scale_key=None,
):
    # logging.info("---- run fused_qkv_rope_ref ----")
    src = src.reshape([total_batch, -1, src.shape[-1]])

    qkv_out = paddle.matmul(src, qkv_weights, False, transpose)
    if qkv_biases is not None:
        qkv_out = paddle.add(qkv_out, qkv_biases)

    fused_hidden_size = qkv_out.shape[2]
    kv_num_heads = (fused_hidden_size - num_head * head_dim) // head_dim // 2
    num_groups = num_head // kv_num_heads
    target_shape = [0, 0, (num_groups + 2) * kv_num_heads, head_dim]

    qkv_out = paddle.reshape_(qkv_out, target_shape)

    query_states, key_states, value_states = paddle.split(
        qkv_out,
        num_or_sections=[num_head, kv_num_heads, kv_num_heads],
        axis=2,
    )

    cos, sin = rotary_embs[0], rotary_embs[1]

    query_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
        query_states,
        None,
        None,
        sin=sin,
        cos=cos,
        use_neox_rotary_style=use_neox_style,
    )
    key_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
        key_states,
        None,
        None,
        sin=sin,
        cos=cos,
        use_neox_rotary_style=use_neox_style,
    )
    key_value_states = paddle.stack([key_states, value_states], axis=0)

    if measurement_mode:
        qkv_act_amax = paddle.max(paddle.abs(src))
        q_amax = paddle.max(paddle.abs(query_states))
        k_amax = paddle.max(paddle.abs(key_states))
        v_amax = paddle.max(paddle.abs(value_states))
        q_scale_key = qkv_act_scale_key.replace("qkv_proj", "q_matmul")
        k_scale_key = qkv_act_scale_key.replace("qkv_proj", "cachek_matmul")
        v_scale_key = qkv_act_scale_key.replace("qkv_proj", "cachev_matmul")
        measure_matrix(qkv_act_amax, qkv_act_scale_key)
        measure_matrix(q_amax, q_scale_key)
        measure_matrix(k_amax, k_scale_key)
        measure_matrix(v_amax, v_scale_key)

    return (
        query_states,
        key_value_states,
    )


def fused_sdpa_ref(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attn_bias: paddle.Tensor,
    is_causal: bool,
    scale: float,
    measurement_mode: bool = False,
) -> paddle.Tensor:
    _, _, query_heads, _ = query.shape
    _, _, kv_heads, _ = key.shape

    query = query.transpose([0, 2, 1, 3])
    key = key.transpose([0, 2, 1, 3])
    value = value.transpose([0, 2, 1, 3])

    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(2)

        attn_weights = paddle.matmul(query, key.transpose([0, 1, 2, 4, 3])) * scale
    else:
        attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * scale
    if attn_bias is not None:
        attn_weights.add_(attn_bias)
    elif is_causal:
        attn_bias = paddle.triu(paddle.ones_like(attn_weights) * -1e4, 1).astype(
            attn_weights.dtype
        )
        attn_weights.add_(attn_bias)
    attn_weights_fused = paddle.nn.functional.softmax(attn_weights, axis=-1)

    # Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    max_score = paddle.max(attn_weights, axis=-1, keepdim=True)
    attn_weights_steps = attn_weights - max_score
    attn_weights_steps = paddle.exp(attn_weights_steps)
    sum_exp = paddle.sum(attn_weights_steps, axis=-1, keepdim=True)
    attn_weights_steps_final = attn_weights_steps / sum_exp

    attn_weights = attn_weights_steps_final

    if measurement_mode:
        s_amax = paddle.max(paddle.abs(attn_weights))
    attn_weights = paddle.matmul(attn_weights, value)

    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)

    attn_weights = attn_weights.transpose([0, 2, 1, 3])
    return attn_weights, s_amax if measurement_mode else attn_weights


def is_gqa(q, k):
    gqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[2]
        kv_heads = k.shape[2]
        gqa = (q_heads != kv_heads) and kv_heads != 1
    return gqa


def gqa_input_reshape_fwd(q, k, v):
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    q_heads_per_group = q_heads // kv_heads

    k = k.repeat_interleave(q_heads_per_group, axis=2)
    v = v.repeat_interleave(q_heads_per_group, axis=2)

    return k, v


def fused_sdpa_proj_ref(
    query_states,
    key_value_states,
    attention_mask,
    linear_weights,
    scaling_factor,
    causal=True,
    softmax_mode="None",
    measurement_mode=False,
    o_act_scale_key=None,
):
    # logging.info("---- run fused_sdpa_proj_ref ----")
    bsz, q_len, num_heads, head_dim = query_states.shape
    key_states = key_value_states[0]
    value_states = key_value_states[1]

    use_fsdpa = False

    if use_fsdpa:
        if is_gqa(query_states, key_states):
            key_states, value_states = gqa_input_reshape_fwd(
                query_states, key_states, value_states
            )

        if measurement_mode:
            attn_output, s_amax = paddlenlp_ops.fused_fp8_sdpa(
                query_states,
                key_states,
                value_states,
                attention_mask,
                None,
                None,
                None,
                None,
                None,
                None,
                causal,
                scaling_factor,
                is_amax_s=True,
            )
        else:
            attn_output, _ = paddlenlp_ops.fused_fp8_sdpa(
                query_states,
                key_states,
                value_states,
                attention_mask,
                None,
                None,
                None,
                None,
                None,
                None,
                causal,
                scaling_factor,
                is_amax_s=False,
            )
        """
        attn_output = paddle.incubate.nn.functional.fused_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            0.0,
            causal,
            scaling_factor,
            False, # is_training
        )
        """
    else:
        if measurement_mode:
            attn_output, s_amax = fused_sdpa_ref(
                query_states,
                key_states,
                value_states,
                attention_mask,
                causal,
                scaling_factor,
                measurement_mode=measurement_mode,
            )
        else:
            attn_output = fused_sdpa_ref(
                query_states,
                key_states,
                value_states,
                attention_mask,
                causal,
                scaling_factor,
                measurement_mode=measurement_mode,
            )
    attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])

    if measurement_mode:
        o_amax = paddle.max(paddle.abs(attn_output))
        s_scale_key = o_act_scale_key.replace("o_proj", "s_matmul")
        measure_matrix(s_amax, s_scale_key)
        measure_matrix(o_amax, o_act_scale_key)
    out_linear_out = paddle.matmul(attn_output, linear_weights)

    return out_linear_out


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
    measurement_mode=False,
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

    if measurement_mode:
        q_scaling_amax = paddle.max(paddle.abs(query))
    attn = paddle.matmul(query, key)

    # if 'fp32_softmax' in enabled_flags():
    #     attn = attn.float()
    attn = attn + block_bias

    block_max = attn.max(axis=-1, keepdim=True)
    adjustment_target_shape = block_max.shape
    attn = attn.subtract(block_max)
    attn = attn.exp()
    # attn = attn.to(value.dtype)
    if measurement_mode:
        s_amax = paddle.max(paddle.abs(attn))

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

    if measurement_mode:
        o_amax = paddle.max(paddle.abs(attn))
    res = paddle.matmul(attn.view([batch_size, hidden_size]), linear_weights)

    return (res, q_scaling_amax, s_amax, o_amax) if measurement_mode else res


def fused_block_attention_ref(
    src,
    rotary_embs,
    k_cache,
    v_cache,
    block_groups,
    block_list,
    block_mapping,
    block_bias,
    block_indices,
    block_offsets,
    qkv_weights,
    qkv_biases,
    out_weights,
    head_dim,
    num_heads,
    scaling_factor,
    transpose=False,
    use_neox_style=False,
    measurement_mode=False,
    qkv_act_scale_key=None,
    o_act_scale_key=None,
):
    # logging.info("---- run fused_block_attention_ref ----")
    query_states, key_value_states = paddlenlp_ops.fused_qkv_rope(
        src,
        qkv_weights,
        qkv_biases,
        rotary_embs.unsqueeze(2),
        None,  # act_scale
        None,  # weight_scale
        None,  # q_scale
        None,  # cache_k_scale
        None,  # cache_v_scale
        None,  # q_norm
        None,  # k_norm
        head_dim,
        num_heads,
        total_batch=src.shape[0],
        transpose=transpose,
        use_neox_style=use_neox_style,
        epsilon=1e-6,
    )
    key_states = key_value_states[0].squeeze(1)
    value_states = key_value_states[1].squeeze(1)
    k_cache.index_put_((block_indices, block_offsets), key_states)
    v_cache.index_put_((block_indices, block_offsets), value_states)
    if measurement_mode:
        qkv_act_amax = paddle.max(paddle.abs(src))
        q_amax = paddle.max(paddle.abs(query_states))
        k_amax = paddle.max(paddle.abs(key_states))
        v_amax = paddle.max(paddle.abs(value_states))
        out_linear_out_ref, _, s_amax, o_amax = fused_flatpa_proj_ref(
            query_states,
            k_cache,
            v_cache,
            block_groups,
            block_list,
            block_mapping,
            block_bias,
            out_weights,
            scaling_factor,
            measurement_mode,
        )
        q_scale_key = qkv_act_scale_key.replace("qkv_proj", "q_matmul")
        k_scale_key = qkv_act_scale_key.replace("qkv_proj", "cachek_matmul")
        v_scale_key = qkv_act_scale_key.replace("qkv_proj", "cachev_matmul")
        s_scale_key = qkv_act_scale_key.replace("qkv_proj", "s_matmul")
        measure_matrix(qkv_act_amax, qkv_act_scale_key)
        measure_matrix(q_amax, q_scale_key)
        measure_matrix(k_amax, k_scale_key)
        measure_matrix(v_amax, v_scale_key)
        measure_matrix(s_amax, s_scale_key)
        measure_matrix(o_amax, o_act_scale_key)
    else:
        out_linear_out_ref = fused_flatpa_proj_ref(
            query_states,
            k_cache,
            v_cache,
            block_groups,
            block_list,
            block_mapping,
            block_bias,
            out_weights,
            scaling_factor,
        )
    return out_linear_out_ref


def fused_mlp_ref(
    hidden_states,
    proj_weight,
    up_weight,
    down_weight,
    permuted_weights,
    measurement_mode=False,
    up_gate_act_scale_key=None,
    down_act_scale_key=None,
):
    # logging.info("---- run fused_mlp_ref ----")
    def swiglu_naive(hidden_states, up=None):
        if up is not None:
            gate = hidden_states
        else:
            gate, up = paddle.chunk(hidden_states, chunks=2, axis=-1)
        silu = gate / (paddle.exp(-gate) + 1)
        return silu * up

    if measurement_mode:
        amax = paddle.max(paddle.abs(hidden_states))
        measure_matrix(amax, up_gate_act_scale_key)
    gate = paddle.matmul(hidden_states, proj_weight, transpose_y=permuted_weights)
    up = (
        paddle.matmul(hidden_states, up_weight, transpose_y=permuted_weights)
        if up_weight is not None
        else None
    )
    swiglu = swiglu_naive(hidden_states=gate, up=up)
    if measurement_mode:
        amax = paddle.max(paddle.abs(swiglu))
        measure_matrix(amax, down_act_scale_key)
    res = paddle.matmul(swiglu, down_weight, transpose_y=permuted_weights)

    return res


def fused_gate_moe_ref(
    hidden_states,
    gate_weights,
    gate_correction_bias,
    up_gate_weights,
    down_weights,
    top_k,
    norm_topk_prob,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    chunk_size,
    measurement_mode=False,
    up_gate_act_scale_key=None,
    down_act_scale_key=None,
):
    # logging.info("---- run fused_gate_moe_ref ----")
    gate_out = paddle.matmul(hidden_states.cast("float32"), gate_weights)
    weights = paddle.nn.functional.softmax(gate_out, axis=-1)
    if gate_correction_bias is not None:
        scores = weights + gate_correction_bias
        _, selected_experts = paddle.topk(scores, top_k, axis=-1)
        routing_weights = paddle.index_sample(weights, selected_experts)
    else:
        routing_weights, selected_experts = paddle.topk(weights, top_k, axis=-1)
    if norm_topk_prob:
        routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)
    routing_weights = routing_weights.cast("bfloat16")
    common_inputs = (hidden_states, selected_experts, routing_weights)

    num_experts = up_gate_weights.shape[0]
    up_gate_proj_weight = [up_gate_weights[i] for i in range(num_experts)]
    down_proj_weight = [down_weights[i] for i in range(num_experts)]

    weights = (up_gate_proj_weight, down_proj_weight)
    common_params = (
        permuted_weights,
        activation,  # "silu",
        experts_min,
        experts_max,
        measurement_mode,
        chunk_size,
    )
    fused_moe_out, amax_per_expert = paddlenlp_ops.mixture_of_experts(
        *common_inputs, *weights, *common_params
    )
    if measurement_mode:
        amax = paddle.max(paddle.abs(hidden_states))
        measure_matrix(amax, up_gate_act_scale_key)
        measure_matrix(amax_per_expert, down_act_scale_key, experts_min, experts_max)
    return fused_moe_out
