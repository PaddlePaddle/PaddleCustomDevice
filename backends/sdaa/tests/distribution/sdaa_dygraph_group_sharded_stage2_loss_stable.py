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


import gc
import numpy as np

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

from paddle.nn import Linear

seed = 2022
epoch = 3

np.random.seed(seed)
paddle.seed(seed)


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=1000, param_attr=None, bias_attr=None):
        super().__init__()
        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=2000, linear_size=1000):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype("float32")
        label = np.ones(1).astype("int64")
        return img, label

    def __len__(self):
        return self.num_samples


def optimizer_setting(model, use_pure_fp16, opt_group=False):
    # TODO(qiulj):support ClipGradByGlobalNorm fp16 + multi_precision
    clip = None
    if not use_pure_fp16:
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        parameters=[
            {
                "params": model.parameters(),
            }
        ]
        if opt_group
        else model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=clip,
        multi_precision=False,
    )

    return optimizer


def train_mlp(
    model,
    sharding_stage,
    batch_size=100,
    use_pure_fp16=False,
    accumulate_grad=False,
    opt_group=False,
    test_minimize=False,
):
    scaler = paddle.amp.GradScaler(
        init_loss_scaling=65536, use_dynamic_loss_scaling=False
    )
    # TODO(qiulj):support opt_group
    # if opt_group:
    #     optimizer = optimizer_setting(
    #         model=model, use_pure_fp16=use_pure_fp16, opt_group=opt_group
    #     )
    # else:
    optimizer = optimizer_setting(model=model, use_pure_fp16=use_pure_fp16)

    if sharding_stage == 2:
        group = paddle.distributed.new_group(
            list(range(paddle.distributed.get_world_size())), backend="xccl"
        )

        optimizer = GroupShardedOptimizerStage2(
            params=optimizer._parameter_list,
            optim=optimizer,
            group=group,
            device="sdaa",
        )

        model = GroupShardedStage2(
            model, optimizer, group=group, buffer_max_size=2**21, device="sdaa"
        )
        scaler = GroupShardedScaler(scaler)
    else:
        model = paddle.DataParallel(model)

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    if sharding_stage == 2:
        model.to(device="sdaa")

    losses = []
    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            with paddle.amp.auto_cast(use_pure_fp16, level="O1"):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(input=out, label=label)

            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))
            if batch_size == 20:
                avg_loss = avg_loss / 5
            if use_pure_fp16:
                scaler.scale(avg_loss).backward()
            else:
                avg_loss.backward()

            losses.append(avg_loss)

            if not accumulate_grad:
                if not use_pure_fp16:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.clear_grad()

        if accumulate_grad:
            if not use_pure_fp16:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.clear_grad()

    # release group before return
    if sharding_stage == 2:
        del model._group
        del optimizer._group
        del optimizer._optim._grad_clip
        gc.collect()

    return losses


def test_sharding_stage2_loss_stable():
    paddle.distributed.init_parallel_env()
    mlp = MLP()
    state_dict = mlp.state_dict()
    mlp1 = MLP()
    mlp2 = MLP()
    mlp3 = MLP()
    mlp4 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)

    losses1 = train_mlp(mlp1, sharding_stage=2)
    losses2 = train_mlp(mlp2, sharding_stage=2)

    # stage2
    for i in range(0, len(losses1)):
        loss1 = losses1[i].numpy()
        loss2 = losses2[i].numpy()
        np.testing.assert_equal(loss1, loss2)

    losses3 = train_mlp(mlp3, sharding_stage=2, accumulate_grad=True)
    losses4 = train_mlp(mlp4, sharding_stage=2, accumulate_grad=True)

    # stage2 accumulate grad
    for i in range(0, len(losses3)):
        loss3 = losses3[i].numpy()
        loss4 = losses4[i].numpy()
        np.testing.assert_equal(loss3, loss4)


if __name__ == "__main__":
    test_sharding_stage2_loss_stable()
