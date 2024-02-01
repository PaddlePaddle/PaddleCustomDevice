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

from __future__ import print_function

import unittest
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)
import numpy as np

from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle

from npu_utils import check_soc_version


def gather_nd_grad(x, index):
    dout_shape = index.shape[:-1] + x.shape[index.shape[-1] :]
    numel = 1
    for i in dout_shape:
        numel = numel * i
    dout = np.full(dout_shape, 1.0 / numel)
    dx = np.full_like(x, 0)

    index = tuple(index.reshape(-1, index.shape[-1]).T)
    np.add.at(dx, index, dout)

    return dx


class TestGatherNdOpWithEmptyIndex(OpTest):
    # Index has empty element, which means copy entire tensor

    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", select_npu)
        self.op_type = "gather_nd"
        xnp = convert_float_to_uint16(
            np.random.uniform(0, 100, (10, 10)).astype(np.float32)
        )
        index = np.array([[1], [2]]).astype("int64")

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": convert_uint16_to_float(xnp[tuple(index.T)])}
        self.x_grad = convert_uint16_to_float(gather_nd_grad(xnp, index))

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
