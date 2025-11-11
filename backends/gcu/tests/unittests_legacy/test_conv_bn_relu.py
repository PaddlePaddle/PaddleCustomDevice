# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.static as static
import paddle.nn.functional as F

paddle.enable_static()

x = static.data(name="x", shape=[4, 3, 224, 224], dtype="float32")
conv = static.nn.conv2d(
    input=x,
    num_filters=64,
    filter_size=7,
    stride=[1, 1],
    padding=[3, 3],
    bias_attr=False,
)
bn = static.nn.batch_norm(input=conv)
out = F.hardswish(bn)

place = paddle.CPUPlace()
exe = static.Executor(place)
exe.run(static.default_startup_program())
prog = static.default_main_program()

static.save_inference_model("./conv_bn_hard_swish", [x], [out], exe)
