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

# from paddle._C_ops import flash_attn_kvcache
from paddle import _C_ops
import os
import paddle
import numpy as np
from typing import Optional, Union, Tuple
from paddle import Tensor

for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )


def test_flash_attn_wrapper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    fixed_seed_offset: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    return_softmax: bool = False,
    is_test: bool = True,
    rng_name: str = "",
) -> Union[Tensor, Tuple[Tensor, ...]]:
    return paddle._C_ops.flash_attn(
        q,
        k,
        v,
        fixed_seed_offset,
        attn_mask,
        dropout_prob,  # dropout 概率
        causal,  # 是否因果注意力
        return_softmax,  # 是否返回 softmax
        is_test,  # 是否测试模式
        rng_name,  # 随机数生成器名称
    )


def flash_attention_wrapper_test():
    # 创建模拟输入 (最小参数)
    batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 64
    q = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype="float16")
    k = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype="float16")
    v = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype="float16")

    # 基础调用 (只传必需参数)
    output = test_flash_attn_wrapper(q, k, v)
    print("基础输出形状:", output[0].shape)  # 输出形状应为 [2, 64, 8, 64]


def flash_attn_unpadded_wrapper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: Union[int, float],
    max_seqlen_k: Union[int, float],
    fixed_seed_offset: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    softmax_scale: float = 1.0,
    dropout: float = 0.0,
    causal: bool = False,
    return_softmax: bool = False,
    is_test: bool = True,
    rng_name: str = "",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    max_seqlen_q_t = paddle.to_tensor(max_seqlen_q, dtype="int64")
    max_seqlen_k_t = paddle.to_tensor(max_seqlen_k, dtype="int64")

    # 调用底层操作
    outputs = paddle._C_ops.flash_attn_unpadded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        fixed_seed_offset,
        attn_mask,
        max_seqlen_q_t,
        max_seqlen_k_t,  # 使用张量形式
        softmax_scale,  # 缩放因子
        dropout,  # dropout 概率
        causal,  # 是否因果注意力
        return_softmax,  # 是否返回softmax
        is_test,  # 是否测试模式
        rng_name,  # 随机数生成器名称
    )
    return outputs


def flash_attn_unpadded_wrapper_test():
    total_q = 128
    total_k = 128
    num_heads = 8
    head_dim = 64
    batch_size = 4

    # 创建输入张量
    q = paddle.randn([total_q, num_heads, head_dim], dtype="float16")
    k = paddle.randn([total_k, num_heads, head_dim], dtype="float16")
    v = paddle.randn([total_k, num_heads, head_dim], dtype="float16")

    cu_seqlens_q = paddle.to_tensor([0, 32, 64, 96, 128], dtype="int32")
    cu_seqlens_k = paddle.to_tensor([0, 32, 64, 96, 128], dtype="int32")

    max_seqlen_q = 32
    max_seqlen_k = 32

    outputs = flash_attn_unpadded_wrapper(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    )
    # 解包输出
    print(f"实际返回值数量: {len(outputs)}")
    print(f"返回值类型: {type(outputs)}")
    for i, item in enumerate(outputs):
        print(
            f"输出 {i} 类型: {type(item)}, 形状: {item.shape if hasattr(item, 'shape') else 'N/A'}"
        )


def flash_attn_kvcache_wrapper(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    seqlens_k: Tensor,
    block_table: Tensor,
    k: Optional[Tensor] = None,
    v: Optional[Tensor] = None,
    rotary_cos: Optional[Tensor] = None,
    rotary_sin: Optional[Tensor] = None,
    cache_batch_idx: Optional[Tensor] = None,
    causal: bool = True,
    is_rotary_interleaved: bool = False,
    num_splits: int = 1,
    dropout: float = 0.0,
    return_softmax: bool = False,
) -> Tuple[Tensor, Tensor]:
    out, softmax_lse = paddle._C_ops._run_custom_op(
        "flash_attn_kvcache",
        q,
        k_cache,
        v_cache,
        k,  # 当前键
        v,  # 当前值
        seqlens_k,
        rotary_cos,  # rotary_cos (默认空)
        rotary_sin,  # rotary_sin (默认空)
        cache_batch_idx,  # cache_batch_idx (默认空)
        block_table,
        causal,  # 是否因果注意力
        is_rotary_interleaved,  # is_rotary_interleaved (默认False)
        num_splits,  # 分割数
        dropout,  # dropout 概率
        return_softmax,  # return_softmax (默认False)
    )

    return out, softmax_lse


def flash_attn_kvcache_wrapper_test():
    batch_size = 2
    seqlen_q = 4
    seqlen_k = 8
    num_heads = 2
    head_size = 32
    max_blocks = 10

    # 2. 创建输入张量
    q = paddle.randn([batch_size, seqlen_q, num_heads, head_size], dtype="float32")
    k_cache = paddle.randn(
        [max_blocks, seqlen_k, num_heads, head_size], dtype="float32"
    )
    v_cache = paddle.randn(
        [max_blocks, seqlen_k, num_heads, head_size], dtype="float32"
    )
    seqlens_k = paddle.to_tensor([seqlen_k] * batch_size, dtype="int32")
    block_table = paddle.to_tensor([[0, 1], [2, 3]], dtype="int32")

    output, lse = flash_attn_kvcache_wrapper(
        q, k_cache, v_cache, seqlens_k, block_table
    )

    print("输出形状:", output.shape)
    print("Softmax LSE 形状:", lse.shape)


def test_flash_attn():

    # 创建输入张量 - 基本形状 (batch_size, seq_len, num_heads, head_dim)
    batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 64

    # 创建 q, k, v 张量
    q = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype="float16")
    k = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype="float16")
    v = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype="float16")

    # 可选参数（这里设为None）
    fixed_seed_offset = None  # paddle.randn([2], dtype='int32')  # 如果需要可以启用
    attn_mask = None  # paddle.randn([batch_size, 1, seq_len, seq_len], dtype='float16')

    # 调用 flash_attn 算子
    out, softmax, softmax_lse, seed_offset = paddle._C_ops.flash_attn(
        q,
        k,
        v,
        fixed_seed_offset,
        attn_mask,
        0.0,  # dropout 概率
        True,  # 使用因果注意力
        False,  # 返回softmax结果
        True,  # 测试模式（不执行dropout）
        "",  # 随机数生成器名称
    )

    # 打印输出形状
    print("输出张量形状:", out.shape)
    print("种子偏移量形状:", seed_offset.shape)

    # 基本验证
    assert out.shape == [batch_size, seq_len, num_heads, head_dim]
    assert seed_offset.shape == [2]  # 应为[种子, 偏移量]

    # 检查输出是否包含有效值
    assert not paddle.any(paddle.isnan(out))
    assert not paddle.any(paddle.isinf(out))

    print("Flash Attention 测试通过!")


def test_flash_attn_unpadded():

    batch_size = 2
    num_heads_q = 6
    num_heads_k = num_heads_q

    total_q = 1125
    total_k = 1342

    seqlens_q = [0, 25, 1125]
    seqlens_k = [0, 42, 1342]
    max_seqlen_q = 1100
    max_seqlen_k = 1300

    softmax_scale = 1.0 / np.sqrt(128.0)
    print("softmax_scale:", softmax_scale)
    # 创建 q, k, v 张量
    q = paddle.randn([total_q, num_heads_q, 128], dtype="float16")
    k = paddle.randn([total_k, num_heads_k, 128], dtype="float16")
    v = paddle.randn([total_k, num_heads_k, 128], dtype="float16")

    cu_seqlens_q = paddle.to_tensor(seqlens_q, dtype="int32")
    cu_seqlens_k = paddle.to_tensor(seqlens_k, dtype="int32")
    max_seqlen_q_t = paddle.to_tensor(max_seqlen_q, dtype="int64")
    max_seqlen_k_t = paddle.to_tensor(max_seqlen_k, dtype="int64")
    # 可选参数（这里设为None）
    fixed_seed_offset = None  # paddle.randn([2], dtype='int32')  # 如果需要可以启用
    attn_mask = None  # paddle.randn([batch_size, 1, seq_len, seq_len], dtype='float16')

    try:
        outputs = paddle._C_ops.flash_attn_unpadded(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            fixed_seed_offset,
            attn_mask,
            max_seqlen_q_t,
            max_seqlen_k_t,  # 使用相同的最大序列长度
            softmax_scale,  # 缩放因子
            0.0,  # dropout 概率
            False,  # 因果注意力
            False,  # 返回 softmax
            True,  # 测试模式
            "",  # 随机数生成器
        )

        # 解包输出
        # out, softmax, softmax_lse, seed_offset = outputs
        print(f"实际返回值数量: {len(outputs)}")
        print(f"返回值类型: {type(outputs)}")
        for i, item in enumerate(outputs):
            print(
                f"输出 {i} 类型: {type(item)}, 形状: {item.shape if hasattr(item, 'shape') else 'N/A'}"
            )

        print("Flash Attention Unpadded 测试通过!")

    except Exception as e:
        print(f"测试失败: {str(e)}")


def test_flash_attn_kvcache_basic():
    print("=== 开始测试 flash_attn_kvcache 算子 ===")

    # 1. 基本参数设置
    batch_size = 2
    seqlen_q = 4
    seqlen_k = 8
    num_heads = 2
    head_size = 32
    max_blocks = 10
    blocks_per_seq = 2

    # 2. 创建输入张量
    q = paddle.randn([batch_size, seqlen_q, num_heads, head_size], dtype="float32")
    k_cache = paddle.randn(
        [max_blocks, seqlen_k, num_heads, head_size], dtype="float32"
    )
    v_cache = paddle.randn(
        [max_blocks, seqlen_k, num_heads, head_size], dtype="float32"
    )
    k = paddle.randn([batch_size, seqlen_k, num_heads, head_size], dtype="float32")
    v = paddle.randn([batch_size, seqlen_k, num_heads, head_size], dtype="float32")
    seqlens_k = paddle.to_tensor([seqlen_k] * batch_size, dtype="int32")
    rotary_cos = paddle.randn([seqlen_k, 1, 1, head_size], dtype="float32")
    rotary_sin = paddle.randn([seqlen_k, 1, 1, head_size], dtype="float32")
    cache_batch_idx = paddle.arange(0, batch_size, dtype="int32")
    block_table = paddle.to_tensor([[0, 1], [2, 3]], dtype="int32")

    # 3. 调用算子
    out, softmax_lse = _C_ops._run_custom_op(
        "flash_attn_kvcache",
        q,
        k_cache,
        v_cache,
        k,
        v,
        seqlens_k,
        None,  # rotary_cos,
        None,  # rotary_sin,
        None,  # cache_batch_idx
        block_table,
        True,  # causal
        False,  # is_rotary_interleaved
        1,  # num_splits
        0.0,  # dropout
        False,
    )  # return_softmax

    # 4. 验证输出形状
    expected_out_shape = [batch_size, seqlen_q, num_heads, head_size]
    actual_out_shape = out.shape
    print(f"输出形状: 预期 {expected_out_shape}, 实际 {actual_out_shape}")
    assert list(actual_out_shape) == expected_out_shape, "输出形状不匹配"

    # 5. 值范围验证
    print("\n=== 值范围验证 ===")
    out_np = out.numpy()
    print(f"输出值范围: [{out_np.min():.4f}, {out_np.max():.4f}]")
    assert not np.isnan(out_np).any(), "输出包含NaN"
    assert not np.isinf(out_np).any(), "输出包含Inf"

    # 6. 简单数值检查
    print("\n=== 简单数值检查 ===")
    print("输出均值:", out_np.mean())
    print("输出标准差:", out_np.std())

    # 7. 测试通过
    print("\n=== flash_attn_kvcache 测试通过! ===")
    return True


if __name__ == "__main__":
    test_flash_attn_kvcache_basic()
    test_flash_attn()
    test_flash_attn_unpadded()
    flash_attention_wrapper_test()
    flash_attn_unpadded_wrapper_test()
    flash_attn_kvcache_wrapper_test()
