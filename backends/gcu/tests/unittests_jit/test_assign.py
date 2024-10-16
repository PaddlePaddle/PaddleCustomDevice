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

from api_base import ApiBase
import paddle
import pytest
import numpy as np


@pytest.mark.assign
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_assign():
    support_types = ["float16", "float32", "float64", "int32", "int64", "bool"]
    for ty in support_types:
        rd_sz_h = np.random.randint(1, 64)
        rd_sz_w = np.random.randint(1, 64)
        test = ApiBase(
            func=paddle.assign,
            feed_names=["X"],
            feed_shapes=[[1, 3, rd_sz_h, rd_sz_w]],
            is_train=False,
            feed_dtypes=[ty],
        )
        x = np.random.rand(1, 3, rd_sz_h, rd_sz_w)
        x = x.astype(ty)
        test.run(feed=[x])
