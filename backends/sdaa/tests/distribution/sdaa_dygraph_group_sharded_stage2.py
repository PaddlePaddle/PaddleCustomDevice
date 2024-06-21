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

import os
import shutil
import tempfile

import gc
import copy
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
epoch = 1

np.random.seed(seed)
paddle.seed(seed)


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=200, param_attr=None, bias_attr=None):
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
    def __init__(self, num_samples=2000, linear_size=200):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype("float32")
        label = np.random.randint(0, 8)
        return img, label

    def __len__(self):
        return self.num_samples


def optimizer_setting(model, use_pure_fp16, opt_group=False):
    clip = None
    multi_precision = False
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    if use_pure_fp16:
        multi_precision = True
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
        multi_precision=multi_precision,
    )

    return optimizer


def train_mlp(
    model,
    sharding_stage,
    batch_size=100,
    use_pure_fp16=False,
    accumulate_grad=False,
    opt_group=False,
    save_model=False,
    test_minimize=False,
):
    scaler = paddle.amp.GradScaler(
        init_loss_scaling=2**16, use_dynamic_loss_scaling=True
    )
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
    if save_model:
        return model, optimizer
    return model.parameters()


def test_dp_stage2():
    paddle.distributed.init_parallel_env()
    mlp = MLP()
    state_dict = mlp.state_dict()
    mlp1 = MLP()
    mlp2 = MLP()
    mlp3 = MLP()
    mlp4 = MLP()
    mlp5 = MLP()
    mlp6 = MLP()
    mlp7 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)
    mlp5.set_state_dict(state_dict)
    mlp6.set_state_dict(state_dict)
    mlp7.set_state_dict(state_dict)

    # DP VS stage2
    dp_params = train_mlp(
        mlp1, sharding_stage="dp", use_pure_fp16=True, opt_group=False
    )
    stage2_params = train_mlp(
        mlp2, sharding_stage=2, use_pure_fp16=True, opt_group=False
    )

    for i in range(len(dp_params)):
        np.testing.assert_allclose(
            dp_params[i].numpy(), stage2_params[i].numpy(), atol=1e-5, rtol=1e-5
        )

    # stage2 accumulate grad
    stage2_params = train_mlp(mlp3, sharding_stage=2, accumulate_grad=True)
    stage2_accumulate_grad = train_mlp(
        mlp4, sharding_stage=2, batch_size=20, accumulate_grad=True
    )

    for i in range(len(stage2_params)):
        np.testing.assert_allclose(
            stage2_params[i].numpy(),
            stage2_accumulate_grad[i].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    # TODO(qiulj):skip this case
    # # stage2 param list VS param group
    # stage2_params = train_mlp(
    #     mlp5, sharding_stage=2, use_pure_fp16=False, opt_group=True
    # )
    # for i in range(len(dp_params)):
    #     np.testing.assert_allclose(
    #         dp_params[i].numpy(), stage2_params[i].numpy(), rtol=1e-2, atol=1e-2
    #     )

    # save/load model
    output_dir = tempfile.mkdtemp()
    model_file = os.path.join(output_dir, "model.pdmodel")
    optimizer_file = os.path.join(output_dir, "model.pdopt")
    model_stage2, optimizer_stage2 = train_mlp(
        mlp6,
        sharding_stage=2,
        use_pure_fp16=True,
        opt_group=False,
        save_model=True,
    )
    # copy for compare
    saved_model = copy.deepcopy(model_stage2.state_dict())
    saved_opt = copy.deepcopy(optimizer_stage2.state_dict())

    paddle.save(model_stage2.state_dict(), model_file)
    paddle.save(optimizer_stage2.state_dict(), optimizer_file)
    m_state_dict = paddle.load(model_file)
    opt_state_dict = paddle.load(optimizer_file)
    model_stage2.set_state_dict(m_state_dict)
    optimizer_stage2.set_state_dict(opt_state_dict)
    # compare
    loaded_model = copy.deepcopy(model_stage2.state_dict())
    loaded_opt = copy.deepcopy(optimizer_stage2.state_dict())
    for key in saved_model:
        np.testing.assert_allclose(
            saved_model[key].numpy(), loaded_model[key].numpy(), rtol=1e-8, atol=1e-8
        )
    for key in saved_opt:
        np.testing.assert_allclose(
            saved_opt[key].numpy(), loaded_opt[key].numpy(), rtol=1e-8, atol=1e-8
        )
    shutil.rmtree(output_dir)
    return


if __name__ == "__main__":
    test_dp_stage2()
