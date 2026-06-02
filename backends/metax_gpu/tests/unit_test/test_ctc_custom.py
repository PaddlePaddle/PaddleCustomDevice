# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F

paddle.set_device("metax_gpu:0")  # 或 custom_device

batch = 4
width = 80
classes = 6625

preds = paddle.randn([batch, width, classes], dtype="float32")

preds = preds.transpose([1, 0, 2])  # -> [T,N,C]

labels = paddle.randint(1, classes, shape=[batch, 25], dtype="int32")

pred_len = paddle.full([batch], width, dtype="int64")
# label_len = paddle.full([batch], 15, dtype="int64")     # 正常
label_len = paddle.full([batch], 16, dtype="int64")  # 报错

loss = F.ctc_loss(preds, labels, pred_len, label_len)

print(loss)
