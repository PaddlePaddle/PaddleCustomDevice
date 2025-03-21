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

# from vllm_hpu_extension.utils import VLLMKVCache
import os

# prompt
# key_cache.shape
# torch.Size([1054, 128, 32, 128])
# key.shape
# torch.Size([2, 128, 32, 128])

# block_indices.shape
# torch.Size([2])
# p block_indices
# tensor([0, 122], device='hpu:0')

# block_offsets
# None

# decode
# p key.shape
# torch.Size([2, 32, 128])

# p block_indices.shape
# torch.Size([2])
# (Pdb) p block_indices
# tensor([6, 8], device='hpu:0')

# p block_offsets.shape
# torch.Size([2])
# p block_offsets
# tensor([  0, 122], device='hpu:0')

# -> key = keys_fetch_func(key_cache, block_list).transpose(1, 2)
# (Pdb) p block_list
# tensor([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0], device='hpu:0', dtype=torch.int32)
# (Pdb) p block_list.shape
# torch.Size([128])
USE_TORCH = False

if USE_TORCH:
    import torch
    import habana_frameworks.torch as htorch

    def insert_or_update_cache(input, cache, block_indices, block_offsets):
        if block_offsets is None:
            cache.index_copy_(0, block_indices, input)
        else:
            cache.index_put_((block_indices, block_offsets), input)

    class VLLMKVCache(torch.nn.Module):
        def __init__(self):
            super(VLLMKVCache, self).__init__()
            self.use_contiguous_pa = (
                os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() == "true"
            )

        def forward(self, input, cache, block_indices, block_offset):
            insert_or_update_cache(input, cache, block_indices, block_offset)
            return cache

        def fetch_from_cache(self, cache, blocks):
            if self.use_contiguous_pa:
                return cache[: blocks.size(0)]
            else:
                return cache.index_select(0, blocks)

    device = "hpu"
    k_cache = VLLMKVCache()

    key_cache = torch.zeros((1054, 128, 32, 128), dtype=torch.bfloat16, device=device)

    if 1:
        key = torch.ones((2, 128, 32, 128), dtype=torch.bfloat16, device=device)
        block_indices = torch.tensor([0, 122], dtype=torch.int64, device=device)
        block_offsets = None
        k_cache(key, key_cache, block_indices, block_offsets)
        print((key_cache[0] == key[0]).all())
        # print((key_cache[122]==key[1]).all())

    if 0:
        key = torch.ones((2, 32, 128), dtype=torch.bfloat16, device=device)
        block_indices = torch.tensor([6, 8], dtype=torch.int64, device=device)
        block_offsets = torch.tensor([0, 122], dtype=torch.int64, device=device)
        k_cache(key, key_cache, block_indices, block_offsets)
        htorch.core.mark_step()
        torch.hpu.synchronize()
        print((key_cache[6][0] == key[0]).all())
        # print((key_cache[8][122]==key[1]).all())

    if 1:
        block_list = torch.tensor(
            [
                0,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=torch.int32,
            device=device,
        )
        key = k_cache.fetch_from_cache(key_cache, block_list)
        print(key[1])

else:
    import paddle
    import paddlenlp_ops

    paddle.set_device("intel_hpu")
    # paddle.set_device("cpu")

    def insert_or_update_cache(input, cache, block_indices, block_offsets):
        if block_offsets is None:
            paddlenlp_ops.index_copy_(cache, block_indices, input, 0)
        else:
            # import pdb; pdb.set_trace()
            # aa = paddle.stack((block_indices, block_offsets), 1)
            cache.index_put_((block_indices, block_offsets), input)

    class VLLMKVCache(paddle.nn.Layer):
        def __init__(self):
            super(VLLMKVCache, self).__init__()
            self.use_contiguous_pa = (
                os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() == "true"
            )

        def forward(self, input, cache, block_indices, block_offset):
            insert_or_update_cache(input, cache, block_indices, block_offset)
            return cache

        def fetch_from_cache(self, cache, blocks):
            if self.use_contiguous_pa:
                return cache[: blocks.shape[0]]
            else:
                return cache.index_select(0, blocks)

    k_cache = VLLMKVCache()

    key_cache = paddle.zeros((1054, 128, 32, 128), dtype="bfloat16")

    if 1:
        key = paddle.ones((2, 128, 32, 128), dtype="bfloat16")
        block_indices = paddle.to_tensor([0, 2], dtype="int64")
        block_offsets = None
        k_cache(key, key_cache, block_indices, block_offsets)
        print((key_cache[0] == key[0]).all())
        # print((key_cache[122]==key[1]).all())

    if 1:
        key_update = paddle.ones((4, 32, 128), dtype="bfloat16")
        block_indices = paddle.to_tensor([6, 8, 10, 2], dtype="int64")
        block_offsets = paddle.to_tensor([0, 122, 144, 2], dtype="int64")
        k_cache(key_update, key_cache, block_indices, block_offsets)
        print((key_cache[6][0] == key[0]).all())
        # print((key_cache[8][122]==key[1]).all())

    if 1:
        block_list = paddle.to_tensor(
            [
                0,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype="int32",
        )
        key_fetch = k_cache.fetch_from_cache(key_cache, block_list)
        print((key_fetch[2] == key[1]).all())
