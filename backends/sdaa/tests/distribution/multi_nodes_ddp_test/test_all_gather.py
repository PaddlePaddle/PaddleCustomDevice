# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import paddle
import paddle.distributed as dist

dist.init_parallel_env()
tensor_list = []
if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
dist.all_gather(tensor_list, data)

print("#" * 30)
print(tensor_list)
print("#" * 30)

# 2nodes-2cards Correct Result:
# [Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[4, 5, 6],
#        [4, 5, 6]]), Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[1, 2, 3],
#        [1, 2, 3]]), Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[1, 2, 3],
#        [1, 2, 3]]), Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[1, 2, 3],
#        [1, 2, 3]]), Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[1, 2, 3],
#        [1, 2, 3]]), Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[1, 2, 3],
#        [1, 2, 3]]), Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[1, 2, 3],
#        [1, 2, 3]]), Tensor(shape=[2, 3], dtype=int64, place=Place(sdaa:0), stop_gradient=True,
#       [[1, 2, 3],
#        [1, 2, 3]])]
