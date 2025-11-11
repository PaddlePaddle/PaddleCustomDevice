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
from paddle.amp.grad_scaler import OptimizerState
from paddle import _C_ops, _legacy_C_ops
from paddle.framework import in_dynamic_mode
import paddle.distributed as dist
from .device_map import *  # noqa


class CustomGradScaler(paddle.amp.GradScaler):
    def __init__(
        self,
        enable=True,
        init_loss_scaling=2.0**16,
        incr_ratio=2.0,
        decr_ratio=0.5,
        incr_every_n_steps=2000,
        decr_every_n_nan_or_inf=1,
        use_dynamic_loss_scaling=True,
    ):
        super().__init__(
            enable,
            init_loss_scaling,
            incr_ratio,
            decr_ratio,
            incr_every_n_steps,
            decr_every_n_nan_or_inf,
            use_dynamic_loss_scaling,
        )
        avaliable_device_list = get_cur_process_device_list()
        group_ranks = [
            avaliable_device_list[0].index(i) for i in avaliable_device_list[1]
        ]
        total_rank = len(group_ranks)
        if paddle.distributed.get_world_size() > 1:
            self.group = paddle.distributed.new_group(group_ranks)
        else:
            self.group = None
        rank = paddle.distributed.get_rank(self.group)
        if rank < 0:
            self.rank = 0
        else:
            self.rank = rank
        self.flatten_grad = None

    def _unscale(self, optimizer):
        if not self._enable:
            return
        optimizer_state = self._optimizer_states[id(optimizer)]
        if optimizer_state["state"] is OptimizerState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["state"] is OptimizerState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")
        if (
            self.group is not None
            and getattr(optimizer, "_rank_param_group", None)
            and in_dynamic_mode()
        ):
            indices = optimizer._rank_param_group[self.rank]
            param_grads_fp32 = []
            temp_tensor = paddle.empty([])
            for param in optimizer._parameter_list:
                if param.stop_gradient or indices.get(param.name) is None:
                    continue
                should_clip, start, end = indices[param.name]
                if param._grad_ivar() is not None:
                    if should_clip:
                        print(self.rank, param.name, start, end)
                        temp_tensor.get_tensor()._share_data_with(
                            param._grad_ivar().get_tensor()
                        )
                        param_grads_fp32.append(
                            temp_tensor.reshape_([-1])._slice(int(start), int(end))
                        )
                    else:
                        param_grads_fp32.append(param._grad_ivar())
            _legacy_C_ops.check_finite_and_unscale(
                param_grads_fp32,
                self._scale,
                param_grads_fp32,
                self._temp_found_inf_fp32,
            )
            self._found_inf = _C_ops.bitwise_or(
                self._found_inf, self._temp_found_inf_fp32
            )
            temp_found_inf = paddle.cast(self._found_inf, dtype=paddle.int32)
            dist.stream.all_reduce(
                temp_found_inf,
                op=dist.ReduceOp.MAX,
                group=self.group,
                use_calc_stream=True,
            )
            self._found_inf = paddle.cast(temp_found_inf, dtype=paddle.bool)
        else:
            super()._unscale(optimizer)
