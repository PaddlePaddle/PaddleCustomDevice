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

dist.init_parallel_env()
rank = dist.get_rank()
size = dist.get_world_size()

if rank == 0:
    out_split_sizes = [2, 1, 1, 0]
elif rank == 1:
    out_split_sizes = [1, 2, 0, 1]
else:
    out_split_sizes = [0, 0, 0, 0]

if rank == 0:
    in_split_sizes = [2, 1, 0, 0]
    in_tensor = paddle.to_tensor([1, 2, 3], dtype="int64")  # 总大小3=2+1+0+0
elif rank == 1:
    in_split_sizes = [1, 2, 0, 0]
    in_tensor = paddle.to_tensor([5, 6, 7], dtype="int64")  # 总大小3=1+2+0+0
elif rank == 2:
    in_split_sizes = [1, 0, 0, 0]
    in_tensor = paddle.to_tensor([9], dtype="int64")  # 总大小1=1+0+0+0
else:
    in_split_sizes = [0, 1, 0, 0]
    in_tensor = paddle.to_tensor([13], dtype="int64")  # 总大小1=0+1+0+0

out_size = sum(out_split_sizes)
out_tensor = paddle.empty([out_size], dtype="int64")

print(f"Rank {rank}: out_split_sizes = {out_split_sizes}")
print(f"Rank {rank}: in_split_sizes = {in_split_sizes}")
print(f"Rank {rank}: in_tensor = {in_tensor.numpy()}")
print(f"Rank {rank}: output size = {out_size}")

task = dist.stream.alltoall_single(
    out_tensor, in_tensor, out_split_sizes, in_split_sizes, sync_op=False
)
task.wait()

out = out_tensor.numpy()
print(out)
