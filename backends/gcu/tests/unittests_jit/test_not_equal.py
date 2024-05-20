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

test = ApiBase(
    func=paddle.not_equal,
    feed_names=["x", "y"],
    feed_shapes=[
        [
            3,
        ],
        [
            3,
        ],
    ],
    is_train=False,
)
np.random.seed(1)


@pytest.mark.not_equal
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_not_equal():
    x = np.array([1, 2, 3]).astype("float32")
    y = np.array([1, 3, 2]).astype("float32")
    test.run(feed=[x, y])


test_not_equal()
