# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# Copyright©2020-2023 Shanghai Biren Technology Co., Ltd. All rights reserved.

import numpy as np
import paddle

x = np.random.uniform(-1, 1, [3, 3]).astype(np.float32)

paddle.set_device("SUPA")
x_supa = paddle.to_tensor(x, stop_gradient=False)
res_supa = paddle.abs(x_supa)
print(res_supa)

paddle.set_device("CPU")
x_cpu = paddle.to_tensor(x, stop_gradient=False)
res_cpu = paddle.abs(x_cpu)
print(res_cpu)
