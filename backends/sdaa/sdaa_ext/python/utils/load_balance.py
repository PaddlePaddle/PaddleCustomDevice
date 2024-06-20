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
import numpy as np


def balance(param_list: list):
    flatten_params = list(filter(lambda param: param.trainable, param_list))
    descend_params = list(
        sorted(flatten_params, key=lambda param: np.prod(param.shape), reverse=True)
    )
    # rank_groups[]
    rank_groups = [[[], 0], [[], 0], [[], 0], [[], 0]]

    def get_smallest_group():
        small_group = (
            rank_groups[0] if rank_groups[0][1] < rank_groups[1][1] else rank_groups[1]
        )
        small_group = (
            rank_groups[2] if rank_groups[2][1] < small_group[1] else small_group
        )
        small_group = (
            rank_groups[3] if rank_groups[3][1] < small_group[1] else small_group
        )
        return small_group

    for param in descend_params:
        group = get_smallest_group()
        group[0].append(param)
        group[1] = group[1] + np.prod(param.shape)

    print(rank_groups[0][1])
    print("+++++" * 5)
    print(rank_groups[1][1])
    print("+++++" * 5)
    print(rank_groups[2][1])
    print("+++++" * 5)
    print(rank_groups[3][1])
    print("+++++" * 5)
    flatten_params = (
        rank_groups[0][0] + rank_groups[1][0] + rank_groups[2][0] + rank_groups[3][0]
    )
    return flatten_params
