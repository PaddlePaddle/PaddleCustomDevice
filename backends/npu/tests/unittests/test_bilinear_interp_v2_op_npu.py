#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import sys

from tests.op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.nn.functional import interpolate
import paddle

paddle.enable_static()


def bilinear_interp_np(input,
                       out_h,
                       out_w,
                       scale_w=0,
                       scale_h=0,
                       out_size=None,
                       actual_shape=None,
                       align_corners=True,
                       align_mode=0,
                       data_layout='NCHW'):
    """bilinear interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for i in range(out_h):
        if (align_mode == 0 and not align_corners):
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)

        h = max(0, h)
        hid = 1 if h < in_h - 1 else 0
        if (align_mode == 0 and not align_corners):
            idx_src_h = max(ratio_h * (i + 0.5) - 0.5, 0)
            h1lambda = idx_src_h - h
        else:
            h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
            if (align_mode == 0 and not align_corners):
                w = int(ratio_w * (j + 0.5) - 0.5)
            else:
                w = int(ratio_w * j)
            w = max(0, w)
            wid = 1 if w < in_w - 1 else 0
            if (align_mode == 0 and not align_corners):
                idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
                w1lambda = idx_src_w - w
            else:
                w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda

            out[:, :, i, j] = h2lambda*(w2lambda*input[:, :, h, w] +
                                        w1lambda*input[:, :, h, w+wid]) + \
                h1lambda*(w2lambda*input[:, :, h+hid, w] +
                          w1lambda*input[:, :, h+hid, w+wid])

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


class TestBilinearInterpOp(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace('ascend', 0)

    def setUp(self):
        self.set_npu()
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "bilinear_interp_v2"
        input_np = np.random.random(self.input_shape).astype(self.dtype)

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]
        scale_h = 0
        scale_w = 0
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
                    scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[1]
                scale_h = self.scale[0]
            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = bilinear_interp_np(input_np, out_h, out_w, scale_w, scale_h,
                                       self.out_size, self.actual_shape,
                                       self.align_corners, self.align_mode,
                                       self.data_layout)

        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout
        }
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=self.atol)

    def test_check_grad(self):
        self.__class__.exist_check_grad = True
        if self.dtype == 'float16':
            return
        self.max_relative_error = 0.005
        inputs_to_check = ['X']
        output_names = ['Out']
        no_grad_set = set()
        cpu_place = fluid.CPUPlace()
        cpu_grads = self._get_gradient(inputs_to_check, cpu_place, output_names,
                                       no_grad_set)
        npu_grads = self._get_gradient(inputs_to_check, self.place,
                                       output_names, no_grad_set)
        self._assert_is_close(cpu_grads, npu_grads, inputs_to_check,
                              self.max_relative_error,
                              "Gradient Check between places")

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.5
        self.align_corners = False
        self.align_mode = 1
        self.dtype = 'float32'
        self.atol = 1e-5


class TestBilinearInterpOpAlignCornerTrue(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpOpAlignCornerTrue, self).init_test_case()
        self.align_corners = True


class TestBilinearInterpCaseFP16(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCaseFP16, self).init_test_case()
        self.dtype = 'float16'
        self.atol = 1e-2

class TestBilinearInterpCaseFP16AlignCornerTrue(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCaseFP16AlignCornerTrue, self).init_test_case()
        self.dtype = 'float16'
        self.atol = 1e-2
        self.align_corners = True
         

class TestBilinearInterpCase1(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase1, self).init_test_case()
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.


class TestBilinearInterpCase1AlignCornerTrue(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase1AlignCornerTrue, self).init_test_case()
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.align_corners = True


class TestBilinearInterpCase2(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase2, self).init_test_case()
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.


class TestBilinearInterpCase2AlignCornerTrue(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase2AlignCornerTrue, self).init_test_case()
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.align_corners = True


class TestBilinearInterpCase3(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase3, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.


class TestBilinearInterpCase31(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase31, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.align_corners = True


class TestBilinearInterpCase4(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase4, self).init_test_case()
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")


class TestBilinearInterpCase41(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase41, self).init_test_case()
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = True


class TestBilinearInterpCase5(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase5, self).init_test_case()
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")


class TestBilinearInterpCase51(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase51, self).init_test_case()
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")
        self.align_corners = True


class TestBilinearInterpCase6(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase6, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 33]).astype("int32")


class TestBilinearInterpCase61(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase61, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 33]).astype("int32")
        self.align_corners = True


class TestBilinearInterpCase7(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase7, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = [2.0, 0.5]


class TestBilinearInterpCase71(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpCase71, self).init_test_case()
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = [2.0, 0.5]
        self.align_corners = True


class TestBilinearInterpSame(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpSame, self).init_test_case()
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.


class TestBilinearInterpSame1(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpSame1, self).init_test_case()
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.
        self.align_corners = True


class TestBilinearInterpActualShape(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpActualShape, self).init_test_case()
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")


class TestBilinearInterpActualShape1(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpActualShape1, self).init_test_case()
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


class TestBilinearInterpDataLayout(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpDataLayout, self).init_test_case()
        self.input_shape = [2, 5, 5, 3]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.data_layout = "NHWC"


class TestBilinearInterpOtherMethod1(TestBilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 1


class TestBilinearInterpWithMethod2(TestBilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 0


class TestBilinearInterpOtherMethod3(TestBilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = True
        self.align_mode = 1

class TestBilinearInterpWithMethod4(TestBilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = True
        self.align_mode = 0


class TestBilinearInterpScale1(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpScale1, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 2.


class TestBilinearInterpScale2(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpScale2, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.


class TestBilinearInterpScale3(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpScale3, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 2.
        self.align_corners = True


class TestBilinearInterpScale4(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpScale4, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.
        self.align_corners = True


class TestBilinearInterpZero(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpZero, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 0.2
        self.align_mode = 0

class TestBilinearInterpZero1(TestBilinearInterpOp):
    def init_test_case(self):
        super(TestBilinearInterpZero1, self).init_test_case()
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 0.2
        self.align_mode = 0
        self.align_corners = True

if __name__ == "__main__":
    unittest.main()
