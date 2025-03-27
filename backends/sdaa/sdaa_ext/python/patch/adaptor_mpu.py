# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

import os
import paddle
from paddle import _legacy_C_ops
from paddle.distributed import collective
from paddle.framework import LayerHelper, in_dynamic_mode
from paddle.distributed.fleet.base import topology as tp


def _c_softmax_with_cross_entropy(
    logits,
    label,
    group=None,
    return_softmax=False,
    ignore_index=-100,
):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    global_rank = collective._get_global_env().rank
    rank = global_rank if group is None else group.get_group_rank(global_rank)
    nranks = collective._get_global_env().world_size if group is None else group.nranks

    input_dims = len(list(logits.shape))
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f"Expected input_dims - 1 = label_dims or input_dims == label_dims\
             (got input_dims{input_dims}, label_dims{label_dims})"
        )
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=-1)

    os.environ["ENABLE_PARALLEL_TP"] = "1"
    if in_dynamic_mode():
        softmax, loss = _legacy_C_ops.c_softmax_with_cross_entropy(
            logits,
            label,
            "ring_id",
            ring_id,
            "rank",
            rank,
            "nranks",
            nranks,
            "ignore_index",
            ignore_index,
        )
        if not return_softmax:
            return loss
        else:
            return loss, softmax
    else:
        attrs = {
            "ring_id": ring_id,
            "rank": rank,
            "nranks": nranks,
            "ignore_index": ignore_index,
        }
        helper = LayerHelper("c_softmax_with_cross_entropy", **locals())
        softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
        loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
        helper.append_op(
            type="c_softmax_with_cross_entropy",
            inputs={"Logits": logits, "Label": label},
            outputs={"Softmax": softmax, "Loss": loss},
            attrs=attrs,
        )

        if return_softmax:
            return loss, softmax

        return loss


class ParallelCrossEntropy(paddle.nn.Layer):
    def __init__(self, mp_group=None, name=None, ignore_index=-100):
        super().__init__()
        self.name = name
        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )
        self.ignore_index = ignore_index

    def forward(self, input, label):
        loss = _c_softmax_with_cross_entropy(
            input,
            label,
            group=self.model_parallel_group,
            ignore_index=self.ignore_index,
        )
        return loss


paddle.distributed.fleet.layers.mpu.mp_ops._c_softmax_with_cross_entropy = (
    _c_softmax_with_cross_entropy
)
paddle.distributed.fleet.layers.mpu.mp_layers.ParallelCrossEntropy = (
    ParallelCrossEntropy
)
paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy = ParallelCrossEntropy
paddle.distributed.fleet.layers.mpu.ParallelCrossEntropy = ParallelCrossEntropy
