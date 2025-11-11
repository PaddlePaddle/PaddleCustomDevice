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

import paddle
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import (
    GroupShardedStage3,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
    GroupShardedScaler,
)
from paddle.nn import Linear

epoch = 5
paddle.seed(2022)
np.random.seed(2022)


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=1000, param_attr=None, bias_attr=None):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        # test for trainable & untrainable offload
        self._linear2.weight.stop_gradient = False
        self._linear2.bias.stop_gradient = False
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
    clip = None
    if not use_pure_fp16:
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        parameters=[{"params": model.parameters()}]
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
    use_pure_fp16=False,
    accumulate_grad=False,
    offload=False,
    batch_size=100,
    convert2cpu=False,
):
    group = paddle.distributed.new_group(
        list(range(paddle.distributed.get_world_size()))
    )
    optimizer = optimizer_setting(model=model, use_pure_fp16=use_pure_fp16)

    if use_pure_fp16:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=65536, use_dynamic_loss_scaling=False
        )
        scaler = GroupShardedScaler(scaler)

    model = GroupShardedStage3(
        model,
        optimizer=optimizer,
        group=group,
        offload=offload,
        segment_size=2**15,
        device="sdaa",
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
            with paddle.amp.auto_cast(
                use_pure_fp16,
                level="O1",
                dtype="float16",
            ):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))

            if accumulate_grad:
                avg_loss = avg_loss / 5

            if not use_pure_fp16:
                avg_loss.backward()
            else:
                scaler.scale(avg_loss).backward()

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
    if not convert2cpu:
        model.get_all_parameters()
    else:
        model.get_all_parameters(convert2cpu)
    return model.parameters()


def test_stage3_offload():
    paddle.distributed.init_parallel_env()
    mlp, mlp1, mlp2, mlp3, mlp4, mlp5, mlp6 = (
        MLP(),
        MLP(),
        MLP(),
        MLP(),
        MLP(),
        MLP(),
        MLP(),
    )
    state_dict = mlp.state_dict()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)
    mlp5.set_state_dict(state_dict)
    mlp6.set_state_dict(state_dict)

    # fp32 offload
    stage3_params = train_mlp(mlp1, use_pure_fp16=False)
    stage3_params_offload = train_mlp(mlp2, use_pure_fp16=False, offload=True)
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            stage3_params[i].numpy(),
            stage3_params_offload[i].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

    # # fp16 offload
    stage3_params = train_mlp(mlp3, use_pure_fp16=True)
    stage3_params_offload = train_mlp(mlp4, use_pure_fp16=True, offload=True)
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            stage3_params[i].numpy(),
            stage3_params_offload[i].numpy(),
            rtol=1e-2,
            atol=1e-2,
        )

    # # fp32 accumulate grad offload
    stage3_params = train_mlp(
        mlp5, use_pure_fp16=False, batch_size=20, accumulate_grad=True
    )
    stage3_params_offload = train_mlp(
        mlp6,
        use_pure_fp16=False,
        accumulate_grad=True,
        offload=True,
        batch_size=20,
        convert2cpu=True,
    )
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            stage3_params[i].numpy(),
            stage3_params_offload[i].numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
    return


if __name__ == "__main__":
    test_stage3_offload()
