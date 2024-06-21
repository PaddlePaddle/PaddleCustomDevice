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
import warnings
from paddle_sdaa.sdaa_ext import *  # noqa
import paddle
from ..utils import *  # noqa
from .device_map import *  # noqa
from ..storage import *  # noqa


class DistributeOptimizer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_param = None
        self._rank_param_group = None
        self.clipped_param = {}
        self.amp_o2 = False  # if traversed
        self.groups = []
        self.flatten_params_name = []
        self.group = None
        self.total_rank = None
        self.rank = None
        self.rank_flat_param = None
        self.rank_num = None

    def re_distribution(self):
        avaliable_device_list = get_cur_process_device_list()
        # The strategy only supports in_card sharding
        for devices_list, cur_group_devices, cur_device_id in avaliable_device_list:
            cur_group_ranks = [devices_list.index(i) for i in cur_group_devices]
            self.groups.append(paddle.distributed.new_group(cur_group_ranks))
            if cur_device_id in cur_group_devices:
                self.group = self.groups[-1]
                self.total_rank = len(cur_group_devices)
        rank = paddle.distributed.get_rank(self.group)
        if rank < 0:
            self.rank = 0
        else:
            self.rank = rank

    def re_flatten(self, parameter_list=None):
        if parameter_list is not None:
            warnings.warn("reset opt._parameter_list to args^parameter_list")
            self._parameter_list = parameter_list
        if isinstance(self._parameter_list[0], dict):
            warnings.warn("_parameter_list^type is dict , donnot support flatten")
            return
        self.flatten_params = []
        index_and_padding = {}
        total_num = 0
        # flatten_params = list(
        #     filter(lambda param: param.dtype == "float32",
        #             self._parameter_list))
        self.flatten_params = balance(self._parameter_list)
        for param in self.flatten_params:
            # padding align to 128B, now we assume param is float32, this should change
            numel = (np.prod(param.shape) + 31) // 32 * 32
            index_and_padding[param.name] = (total_num, numel, np.prod(param.shape))
            total_num += numel
        # for all_gather, every rank allocate_flatten_tensor should align 128B, assume param is float32,is 128/4 =32
        align_size = 32 * self.total_rank  # np.lcm(32, self.total_rank)
        if total_num % align_size != 0:
            total_num_ = (total_num + align_size - 1) // align_size * align_size

            self.flatten_params.append(
                paddle.full(
                    shape=[total_num_ - total_num], fill_value=0.0, dtype=paddle.float32
                )
            )
            index_and_padding["align_gap_tensor"] = (
                total_num,
                total_num_ - total_num,
                total_num_ - total_num,
            )
            total_num = total_num_
        self.flat_param = paddle.full(
            shape=[total_num], fill_value=0.0, dtype=paddle.float32
        )
        paddle._legacy_C_ops.coalesce_tensor(
            self.flatten_params,
            self.flatten_params,
            self.flat_param,
            "copy_data",
            True,
            "use_align",
            True,
            "align_size",
            128,
            "dtype",
            self.flatten_params[0].dtype,
        )
        # print(self.flat_param.shape, flush=True)

        self._rank_param_group = {}
        self._rank_pram_gap = {}
        for i in range(self.total_rank):
            self._rank_param_group[i] = {}
            self._rank_pram_gap[i] = {}
        self._rank_pram_gap[self.total_rank] = {}
        rank_num = total_num // self.total_rank
        self.rank_num = rank_num
        # private layout should not be cliped
        should_clip = True
        cur_rank = 0
        cur_rank_allocate_num = 0
        rank_offset = rank_num * self.rank
        self.rank_flat_param = self.flat_param._slice(
            rank_offset, rank_offset + rank_num
        )
        for name, (_, length, numel_) in index_and_padding.items():
            # while True until no params to alloc
            remaining_num = length

            free_num = rank_num - cur_rank_allocate_num

            # if free num is larger than remaining num, fully alloc
            if free_num >= remaining_num:
                self._rank_param_group[cur_rank][name] = (False, 0, 0)
                cur_rank_allocate_num += remaining_num
                # [param,gap_tensor]
                self._rank_pram_gap[cur_rank][name] = numel_
                self._rank_pram_gap[cur_rank][f"{name}_gap"] = length - numel_

                # deal with rank id, if free_num is 0
                if free_num == remaining_num:
                    cur_rank_allocate_num = 0
                    cur_rank += 1

                continue
            # split remaining num, until remaining num is 0
            start_num = 0
            # deal with aligned data
            param_len = numel_
            param_numel = numel_
            while remaining_num > 0:
                free_num = rank_num - cur_rank_allocate_num
                # alloc free num in cur rank
                if remaining_num >= free_num:
                    # extreme case: remaining_num >= free_num >= param_len
                    # it cannot alloc params in the next rank
                    if free_num >= param_len:
                        if param_len != param_numel:
                            self._rank_param_group[cur_rank][name] = (
                                should_clip,
                                start_num,
                                start_num + param_len,
                            )
                        else:
                            self._rank_param_group[cur_rank][name] = (False, 0, 0)
                        self._rank_pram_gap[cur_rank][name] = param_len
                        self._rank_pram_gap[cur_rank][f"{name}_gap"] = (
                            free_num - param_len
                        )
                        cur_rank += 1
                        cur_rank_allocate_num = remaining_num - free_num
                        self._rank_pram_gap[cur_rank][
                            f"{name}_gap"
                        ] = cur_rank_allocate_num
                        break
                    # free num < param_len
                    self._rank_param_group[cur_rank][name] = (
                        should_clip,
                        start_num,
                        start_num + free_num,
                    )
                    # update cur rank id and recompute remaining_num
                    self._rank_pram_gap[cur_rank][name] = free_num
                    start_num += free_num
                    cur_rank += 1
                    cur_rank_allocate_num = 0
                    remaining_num -= free_num
                    param_len -= free_num
                else:  # remaining_num < free_num
                    if param_len != param_numel:
                        self._rank_param_group[cur_rank][name] = (
                            should_clip,
                            start_num,
                            start_num + param_len,
                        )
                    else:
                        self._rank_param_group[cur_rank][name] = (False, 0, 0)
                    self._rank_pram_gap[cur_rank][name] = param_len
                    self._rank_pram_gap[cur_rank][f"{name}_gap"] = (
                        remaining_num - param_len
                    )
                    cur_rank_allocate_num += remaining_num
                    remaining_num = 0
