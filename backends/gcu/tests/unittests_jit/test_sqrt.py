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

test1 = ApiBase(
    func=paddle.sqrt, feed_names=["data"], is_train=False, feed_shapes=[[2, 3]]
)


@pytest.mark.sqrt
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sqrt_1():
    data = np.array([[16, 20, 78], [400, 1, 32]]).astype("float32")
    test1.run(feed=[data])


test2 = ApiBase(
    func=paddle.sqrt, feed_names=["data"], is_train=False, feed_shapes=[[2, 3]]
)


@pytest.mark.sqrt
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sqrt_2():
    data = np.array([[31.4, 1.4, 324.43], [0, 11.11, 22.22]]).astype("float32")
    test2.run(feed=[data])
