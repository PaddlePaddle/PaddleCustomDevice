#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle.base.core as core
import paddle.base as base
import paddle
from paddle.nn.functional import interpolate

paddle.enable_static()


def nearest_neighbor_interp_np(
    X,
    out_h,
    out_w,
    scale_h=0,
    scale_w=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    data_layout="NCHW",
):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        X = np.transpose(X, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    n, c, in_h, in_w = X.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w
    out = np.zeros((n, c, out_h, out_w))

    if align_corners:
        for i in range(out_h):
            in_i = int(ratio_h * i + 0.5)
            for j in range(out_w):
                in_j = int(ratio_w * j + 0.5)
                out[:, :, i, j] = X[:, :, in_i, in_j]
    else:
        for i in range(out_h):
            in_i = int(ratio_h * i)
            for j in range(out_w):
                in_j = int(ratio_w * j)
                out[:, :, i, j] = X[:, :, in_i, in_j]

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    # out = np.expand_dims(out, 2)
    return out.astype(X.dtype)


class TestNearestInterpOpAPI_dy(unittest.TestCase):
    def test_case(self):
        import paddle

        if "gcu" in paddle.base.core.get_all_custom_device_type():
            place = paddle.CustomPlace("gcu", 0)
        else:
            place = core.CPUPlace()
        with base.dygraph.guard(place):
            input_data = np.random.random((2, 3, 6, 6)).astype("float32")
            scale_np = np.array([2, 2]).astype("int64")
            input_x = paddle.to_tensor(input_data)
            scale = paddle.to_tensor(scale_np)
            expect_res = nearest_neighbor_interp_np(
                input_data, out_h=12, out_w=12, align_corners=False
            )
            out = interpolate(
                x=input_x, scale_factor=scale, mode="nearest", align_corners=False
            )
            self.assertTrue(np.allclose(out.numpy(), expect_res))


if __name__ == "__main__":
    unittest.main()
