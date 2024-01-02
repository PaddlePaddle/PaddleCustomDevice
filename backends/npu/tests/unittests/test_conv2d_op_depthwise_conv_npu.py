# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from tests.op_test import OpTest

paddle.enable_static()
SEED = 2021


def conv2d_forward_naive(
    input, filter, group, conv_param, padding_algorithm="EXPLICIT", data_format="NCHW"
):
    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError(
            "Unknown Attr(padding_algorithm): '%s'. "
            "It can only be 'SAME' or 'VALID'." % str(padding_algorithm)
        )

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Unknown Attr(data_format): '%s' ."
            "It can only be 'NCHW' or 'NHWC'." % str(data_format)
        )

    channel_last = data_format == "NHWC"
    if channel_last:
        input = np.transpose(input, [0, 3, 1, 2])

    in_n, in_c, in_h, in_w = input.shape
    f_n, f_c, f_h, f_w = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group

    stride, pad, dilation = (
        conv_param["stride"],
        conv_param["pad"],
        conv_param["dilation"],
    )

    # update pad and dilation
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(
            input_shape, pool_size, pool_stride
        ):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max(
                ((out_size - 1) * stride_size + filter_size - input_size, 0)
            )
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    ksize = filter.shape[2:4]
    if padding_algorithm == "VALID":
        pad = [0, 0, 0, 0]
    elif padding_algorithm == "SAME":
        dilation = [1, 1]
        input_data_shape = input.shape[2:4]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_h_0, pad_h_1 = pad[0], pad[0]
    pad_w_0, pad_w_1 = pad[1], pad[1]
    if len(pad) == 4:
        pad_h_0, pad_h_1 = pad[0], pad[1]
        pad_w_0, pad_w_1 = pad[2], pad[3]
    out_h = 1 + (in_h + pad_h_0 + pad_h_1 - (dilation[0] * (f_h - 1) + 1)) // stride[0]
    out_w = 1 + (in_w + pad_w_0 + pad_w_1 - (dilation[1] * (f_w - 1) + 1)) // stride[1]
    out = np.zeros((out_n, out_c, out_h, out_w))

    d_bolck_h = dilation[0] * (f_h - 1) + 1
    d_bolck_w = dilation[1] * (f_w - 1) + 1

    input_pad = np.pad(
        input,
        ((0, 0), (0, 0), (pad_h_0, pad_h_1), (pad_w_0, pad_w_1)),
        mode="constant",
        constant_values=0,
    )

    filter_dilation = np.zeros((f_n, f_c, d_bolck_h, d_bolck_w))
    filter_dilation[
        :, :, 0 : d_bolck_h : dilation[0], 0 : d_bolck_w : dilation[1]
    ] = filter

    for i in range(out_h):
        for j in range(out_w):
            for g in range(group):
                input_pad_masked = input_pad[
                    :,
                    g * f_c : (g + 1) * f_c,
                    i * stride[0] : i * stride[0] + d_bolck_h,
                    j * stride[1] : j * stride[1] + d_bolck_w,
                ]

                f_sub = filter_dilation[g * sub_f_n : (g + 1) * sub_f_n, :, :, :]
                # sub_f_n == sub_out_c
                for k in range(sub_out_c):
                    # Multiplication of Corresponding Elements, then sum all
                    out[:, g * sub_out_c + k, i, j] = np.sum(
                        input_pad_masked * f_sub[k, :, :, :], axis=(1, 2, 3)
                    )

    if channel_last:
        out = np.transpose(out, [0, 2, 3, 1])

    return out, in_n, out_h, out_w, out_c


def create_test_channel_last_class(parent):
    class TestChannelLastCase(parent):
        def init_data_format(self):
            self.data_format = "NHWC"

        def init_test_case_2(self):
            N, C, H, W = self.input_size
            self.input_size = [N, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


def create_test_padding_SAME_class(parent):
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.pad = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


def create_test_fp16_class(parent):
    class TestFp16Case(parent):
        def init_data_type(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


# TODO(Aganlengzi)
@unittest.skipIf(
    True, "CANN5.0.4 Issue: https://gitee.com/Ascend/modelzoo/issues/I4TBLP"
)
class TestDepthwiseConvNPU(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "depthwise_conv2d"
        self.init_data_format()
        self.init_data_type()
        self.init_test_case()
        self.init_test_case_2()

        conv2d_param = {
            "stride": self.stride,
            "pad": self.pad,
            "dilation": self.dilations,
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

        output, _, _, _, _ = conv2d_forward_naive(
            input, filter, self.groups, conv2d_param, "EXPLICIT", self.data_format
        )

        output = output.astype(self.dtype)

        self.inputs = {
            "Input": OpTest.np_dtype_to_base_dtype(input),
            "Filter": OpTest.np_dtype_to_base_dtype(filter),
        }

        self.attrs = {
            "strides": self.stride,
            "paddings": self.pad,
            "groups": self.groups,
            "dilations": self.dilations,
            "data_format": self.data_format,
        }
        self.outputs = {"Output": output}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        if self.dilations[0] == 1 and self.dilations[1] == 1:
            if self.dtype == np.float16:
                self.check_grad_with_place(
                    self.place, {"Input", "Filter"}, "Output", max_relative_error=0.9
                )
            else:
                self.check_grad_with_place(
                    self.place,
                    {"Input", "Filter"},
                    "Output",
                    max_relative_error=0.03,
                    numeric_place=paddle.CPUPlace(),
                )

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Output",
                no_grad_set=set(["Filter"]),
                max_relative_error=0.9,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Output",
                no_grad_set=set(["Filter"]),
                max_relative_error=0.03,
                numeric_place=paddle.CPUPlace(),
            )

    def test_check_grad_no_input(self):
        if self.dilations[0] == 1 and self.dilations[1] == 1:
            if self.dtype == np.float16:
                self.check_grad_with_place(
                    self.place,
                    ["Filter"],
                    "Output",
                    no_grad_set=set(["Input"]),
                    max_relative_error=0.9,
                )
            else:
                self.check_grad_with_place(
                    self.place,
                    ["Filter"],
                    "Output",
                    no_grad_set=set(["Input"]),
                    max_relative_error=0.03,
                    numeric_place=paddle.CPUPlace(),
                )

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_data_type(self):
        self.dtype = np.float32

    def init_test_case_2(self):
        pass


class TestDepthwiseConvNPU2(TestDepthwiseConvNPU):
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]


class TestDepthwiseConvNPU3(TestDepthwiseConvNPU):
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]


class TestDepthwiseConvNPU4(TestDepthwiseConvNPU):
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]


# TODO(Aganlengzi)
@unittest.skipIf(
    True, "CANN5.0.4 Issue: https://gitee.com/Ascend/modelzoo/issues/I4TBLP"
)
class TestDepthwiseConvNPU_Padding(OpTest):
    def setUp(self):
        self.op_type = "depthwise_conv2d"
        self.dtype = np.float32
        self.set_npu()
        self.init_data_format()
        self.init_data_type()
        self.init_paddings()
        self.init_test_case()
        self.init_test_case_2()

        conv2d_param = {
            "stride": self.stride,
            "pad": self.pad,
            "dilation": self.dilations,
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

        output, _, _, _, _ = conv2d_forward_naive(
            input,
            filter,
            self.groups,
            conv2d_param,
            self.padding_algorithm,
            self.data_format,
        )
        output = output.astype(self.dtype)

        self.inputs = {
            "Input": OpTest.np_dtype_to_base_dtype(input),
            "Filter": OpTest.np_dtype_to_base_dtype(filter),
        }

        self.attrs = {
            "strides": self.stride,
            "paddings": self.pad,
            "padding_algorithm": self.padding_algorithm,
            "groups": self.groups,
            "dilations": self.dilations,
            "data_format": self.data_format,
        }
        self.outputs = {"Output": output}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, {"Input", "Filter"}, "Output", max_relative_error=1.2
            )
        else:
            self.check_grad_with_place(
                self.place,
                {"Input", "Filter"},
                "Output",
                max_relative_error=0.03,
                numeric_place=paddle.CPUPlace(),
            )

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Output",
                max_relative_error=0.7,
                no_grad_set=set(["Filter"]),
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Output",
                max_relative_error=0.03,
                no_grad_set=set(["Filter"]),
                numeric_place=paddle.CPUPlace(),
            )

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["Filter"],
                "Output",
                max_relative_error=0.8,
                no_grad_set=set(["Input"]),
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["Filter"],
                "Output",
                max_relative_error=0.03,
                no_grad_set=set(["Input"]),
                numeric_place=paddle.CPUPlace(),
            )

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_data_type(self):
        self.dtype = np.float32

    def init_paddings(self):
        self.pad = [1, 1, 0, 1]
        self.padding_algorithm = "EXPLICIT"

    def init_test_case_2(self):
        pass


class TestDepthwiseConvNPU2_Padding(TestDepthwiseConvNPU_Padding):
    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [0, 1, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvNPU3_Padding(TestDepthwiseConvNPU_Padding):
    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 2, 3]
        self.padding_algorithm = "EXPLICIT"


# test channel last
create_test_channel_last_class(TestDepthwiseConvNPU)
create_test_channel_last_class(TestDepthwiseConvNPU2)
create_test_channel_last_class(TestDepthwiseConvNPU_Padding)
create_test_channel_last_class(TestDepthwiseConvNPU2_Padding)

# test padding SAME
create_test_padding_SAME_class(TestDepthwiseConvNPU_Padding)
create_test_padding_SAME_class(TestDepthwiseConvNPU2_Padding)
create_test_padding_SAME_class(TestDepthwiseConvNPU3_Padding)

# test padding VALID
create_test_padding_VALID_class(TestDepthwiseConvNPU_Padding)
create_test_padding_VALID_class(TestDepthwiseConvNPU2_Padding)
create_test_padding_VALID_class(TestDepthwiseConvNPU3_Padding)

create_test_fp16_class(TestDepthwiseConvNPU)
create_test_fp16_class(TestDepthwiseConvNPU2)
create_test_fp16_class(TestDepthwiseConvNPU_Padding)
create_test_fp16_class(TestDepthwiseConvNPU2_Padding)
create_test_fp16_class(TestDepthwiseConvNPU3_Padding)

if __name__ == "__main__":
    unittest.main()
