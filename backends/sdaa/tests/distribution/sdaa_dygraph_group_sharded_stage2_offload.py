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
import gc
from sdaa_dygraph_group_sharded_stage2 import MLP, RandomDataset, optimizer_setting

import paddle
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
    GroupShardedScaler,
)

seed = 2021
epoch = 1
batch_size = 100
linear_size = 200

np.random.seed(seed)
paddle.seed(seed)


def train_mlp(model, offload=False, test=False, dp=False):
    optimizer = optimizer_setting(model=model, use_pure_fp16=True)
    # 开启sharding offload的情况下decorate_level必须为'O2'
    # 精度考虑:
    # sdaa若将model_decorate设置成'O2',将会导致后续的scaler + opt模块均为fp16算子。
    # 此时和offload到cpu的参数存在算子dtype的区别(cpu用fp32计算scaler等),因此我们希望以sharding(O1)对齐sharding_offload(O2)
    model_amp_level = "O1"
    init_grad_scale = 16384
    if offload:
        model_amp_level = "O2"
    model = paddle.amp.decorate(
        models=model, level=model_amp_level, save_dtype="float32"
    )
    scaler = paddle.amp.GradScaler(
        init_loss_scaling=init_grad_scale, use_dynamic_loss_scaling=True
    )
    if not dp:
        scaler = GroupShardedScaler(scaler)
        group = paddle.distributed.new_group(
            list(range(paddle.distributed.get_world_size()))
        )
        optimizer = GroupShardedOptimizerStage2(
            params=optimizer._parameter_list,
            optim=optimizer,
            offload=offload,
            group=group,
            device="sdaa",
        )
        model = GroupShardedStage2(
            model, optimizer, buffer_max_size=2**21, device="sdaa"
        )

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True
            # 修改auto_cast等级为O1,使部分算子拥有更高的精度(fp32),O2等级过于激进,可能导致模型不收敛。
            # PS:auto_cast等级独立于model decorate等级,二者不耦合,可以存在一个O1一个O2的情况。
            with paddle.amp.auto_cast(True, level="O1"):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(input=out, label=label)

            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))
            scaler.scale(avg_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()
    if not dp:
        for dtype in optimizer.param_storages:
            for dst_rank, param_storage in optimizer.param_storages[dtype].items():
                param_storage.to(device="sdaa", dtype=dtype)
        del model._group
        del optimizer._group
        del optimizer._optim._grad_clip
        gc.collect()
    return model.parameters()


def test_sharding_stage2_offload():
    paddle.distributed.init_parallel_env()
    mlp = MLP(linear_size)
    state_dict = mlp.state_dict()
    mlp_offload = MLP(linear_size)
    mlp_offload.set_state_dict(state_dict)
    dp_mlp = MLP(linear_size)
    dp_mlp.set_state_dict(state_dict)

    mlp_params = train_mlp(mlp, offload=False)
    mlp_offload_params = train_mlp(mlp_offload, offload=True)
    dp_mlp_params = train_mlp(dp_mlp, offload=False, dp=True)

    for i in range(len(mlp_params)):
        np.testing.assert_allclose(
            mlp_params[i].numpy(),
            mlp_offload_params[i].numpy(),
            rtol=5e-4,
            atol=5e-4,
        )

    for i in range(len(mlp_params)):
        np.testing.assert_allclose(
            mlp_params[i].numpy(),
            dp_mlp_params[i].numpy(),
            rtol=5e-4,
            atol=5e-4,
        )

    for i in range(len(mlp_offload_params)):
        np.testing.assert_allclose(
            mlp_offload_params[i].numpy(),
            dp_mlp_params[i].numpy(),
            rtol=5e-4,
            atol=5e-4,
        )
    return


if __name__ == "__main__":
    test_sharding_stage2_offload()
