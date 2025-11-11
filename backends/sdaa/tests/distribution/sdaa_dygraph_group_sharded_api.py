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

import shutil
import tempfile

import numpy as np
import gc

import paddle
from paddle.distributed.sharding import (
    group_sharded_parallel,
    save_group_sharded_model,
)
from sdaa_dygraph_group_sharded_stage2 import MLP, RandomDataset

epoch = 3
paddle.seed(2022)
np.random.seed(2022)
batch_size = 100


def optimizer_setting(model, use_multi_precision, opt_group=False):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Momentum(
        parameters=[{"params": list(model.parameters())}]
        if opt_group
        else list(model.parameters()),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=clip,
        multi_precision=use_multi_precision,
    )

    return optimizer


def train_mlp(model, shard_level, use_multi_precision, output_dir, amp_level="O1"):
    model = paddle.amp.decorate(models=model, level=amp_level, save_dtype="float32")
    optimizer = optimizer_setting(model=model, use_multi_precision=use_multi_precision)
    scaler = paddle.amp.GradScaler(
        init_loss_scaling=65536, use_dynamic_loss_scaling=True
    )
    if shard_level != "dp":
        group = paddle.distributed.new_group(
            list(range(paddle.distributed.get_world_size()))
        )
        model, optimizer, scaler = group_sharded_parallel(
            model=model,
            optimizer=optimizer,
            level=shard_level,
            scaler=scaler,
            group=group,
        )
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

    for eop in range(epoch):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True
            with paddle.amp.auto_cast(True, level=amp_level):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))
            scaler.scale(avg_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()
    if shard_level != "dp":
        save_group_sharded_model(model, output=output_dir, optimizer=optimizer)
        del model._group
        del optimizer._group
        del optimizer._optim._grad_clip
        gc.collect()
    return model.parameters()


def test_sharding_api():
    paddle.distributed.init_parallel_env()
    mlp, mlp1, mlp2 = MLP(), MLP(), MLP()
    state_dict = mlp.state_dict()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)

    output_dir = tempfile.mkdtemp()

    # AMP
    dp_mlp = MLP()
    mlp3, mlp4 = MLP(), MLP()
    dp_mlp.set_state_dict(state_dict)
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)

    dp_params = train_mlp(
        dp_mlp,
        shard_level="dp",
        use_multi_precision=False,
        output_dir=output_dir,
        amp_level="O1",
    )
    stage2_params = train_mlp(
        mlp3,
        shard_level="os_g",
        use_multi_precision=False,
        output_dir=output_dir,
        amp_level="O1",
    )
    stage3_params = train_mlp(
        mlp4,
        shard_level="p_g_os",
        use_multi_precision=False,
        output_dir=output_dir,
        amp_level="O1",
    )

    for i in range(len(stage2_params)):
        np.testing.assert_allclose(
            dp_params[i].numpy(),
            stage2_params[i].numpy(),
            rtol=1e-4,
            atol=1e-3,
        )
    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            dp_params[i].numpy(),
            stage3_params[i].numpy(),
            rtol=1e-4,
            atol=1e-3,
        )
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    test_sharding_api()
