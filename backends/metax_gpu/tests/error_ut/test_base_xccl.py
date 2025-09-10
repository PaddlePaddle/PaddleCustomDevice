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

# python -m paddle.distributed.launch --gpus=0,1 test_base_xccl.py
import paddle
import paddle.distributed as dist

# os.environ.update({
#    'PADDLE_TRAINER_ID': '0',
#    'PADDLE_TRAINERS_NUM': '1',
#    'PADDLE_CURRENT_ENDPOINT': '172.23.5.214:6170'
# })


dist.init_parallel_env()
tensor_list = []

if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
dist.all_gather(tensor_list, data)
print("allgather:")
print(tensor_list)
# [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)

local_rank = dist.get_rank()
data = None
print("local_rank = ", local_rank)
if local_rank == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
task = dist.stream.all_reduce(data, sync_op=False)
task.wait()
out = data
print("all_reduce:")
print(out)
# [[5, 7, 9], [5, 7, 9]]:


if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
dist.broadcast(data, src=1)
print("broadcast:")
print(data)
# [[1, 2, 3], [1, 2, 3]] (2 GPUs)


if dist.get_rank() == 0:
    data = paddle.to_tensor([7, 8, 9])
    dist.send(data, dst=1)
else:
    data = paddle.to_tensor([1, 2, 3])
    dist.recv(data, src=0)
print("recv:")
print(data)
# [7, 8, 9] (2 GPUs)


if dist.get_rank() == 0:
    data1 = paddle.to_tensor([7, 8, 9])
    data2 = paddle.to_tensor([10, 11, 12])
    dist.scatter(data1, src=1)
else:
    data1 = paddle.to_tensor([1, 2, 3])
    data2 = paddle.to_tensor([4, 5, 6])
    dist.scatter(data1, tensor_list=[data1, data2], src=1)
print("scatter:")
print(data1, data2)
# [1, 2, 3] [10, 11, 12] (2 GPUs, out for rank 0)
# [4, 5, 6] [4, 5, 6] (2 GPUs, out for rank 1)

if dist.get_rank() == 0:
    data = paddle.to_tensor([7, 8, 9])
    dist.send(data, dst=1)
else:
    data = paddle.to_tensor([1, 2, 3])
    dist.recv(data, src=0)
print("send:")
print(data)
# [7, 8, 9] (2 GPUs)


if dist.get_rank() == 0:
    data1 = paddle.to_tensor([0, 1])
    data2 = paddle.to_tensor([2, 3])
else:
    data1 = paddle.to_tensor([4, 5])
    data2 = paddle.to_tensor([6, 7])
dist.reduce_scatter(data1, [data1, data2])
print("reduce_scatter:")
print(data1)
# [4, 6] (2 GPUs, out for rank 0)
# [8, 10] (2 GPUs, out for rank 1)

out_tensor_list = []
if dist.get_rank() == 0:
    data1 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
    data2 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]])
else:
    data1 = paddle.to_tensor([[13, 14, 15], [16, 17, 18]])
    data2 = paddle.to_tensor([[19, 20, 21], [22, 23, 24]])
dist.alltoall(in_tensor_list=[data1, data2], out_tensor_list=out_tensor_list)
print("alltoall:")
print(out_tensor_list)
# [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]] (2 GPUs, out for rank 0)
# [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]] (2 GPUs, out for rank 1)
