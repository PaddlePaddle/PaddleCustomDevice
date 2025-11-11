# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function

import numpy as np
import unittest

from op_test import OpTest
from paddle import _C_ops
import paddle

paddle.enable_static()
SEED = 2022

from test_conv_op_sdaa import (
    create_test_padding_SAME_class,
    create_test_padding_VALID_class,
    create_test_channel_last_class,
    conv2d_forward_naive,
    create_test_fp16_class,
)

paddle.enable_static()
SEED = 2022


def depthwise_conv2d_wrapper(
    x,
    weight,
    stride=1,
    padding=0,
    padding_algorithm="EXPLICIT",
    groups=1,
    dilation=1,
    data_format="NCDHW",
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    if padding_algorithm is None:
        padding_algorithm = "EXPLICIT"
    return paddle._C_ops.depthwise_conv2d(
        x,
        weight,
        stride,
        padding,
        padding_algorithm,
        groups,
        dilation,
        data_format,
    )


class TestDepthwiseConv2DOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float16

    def init_data_format(self):
        self.data_format = "NCHW"

    def setUp(self):
        self.set_sdaa()
        self.op_type = "depthwise_conv2d"
        self.python_api = depthwise_conv2d_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_data_format()
        self.init_dtype()
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_paddings()

        conv2d_param = {
            "stride": self.stride,
            "pad": self.pad,
            "dilation": self.dilations,
        }
        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)
        output, _, _, _, _ = conv2d_forward_naive(
            input, filter, self.groups, conv2d_param, data_format=self.data_format
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

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
            atol=1e-4,
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            {"Input", "Filter"},
            "Output",
            max_relative_error=0.03,
            numeric_place=paddle.CPUPlace(),
        )

    def test_check_grad_no_filter(self):
        self.check_grad_with_place(
            self.place,
            ["Input"],
            "Output",
            max_relative_error=0.03,
            no_grad_set=set(["Filter"]),
            numeric_place=paddle.CPUPlace(),
        )

    def test_check_grad_no_input(self):
        self.check_grad_with_place(
            self.place,
            ["Filter"],
            "Output",
            max_relative_error=0.03,
            no_grad_set=set(["Input"]),
            numeric_place=paddle.CPUPlace(),
        )

    def init_test_case(self):
        self.pad = [1, 2]
        self.stride = [2, 1]
        self.input_size = [8, 4, 3, 96]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 5, 5]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 4

    def init_paddings(self):
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv2(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [8, 4, 3, 96]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 5, 5]


class TestDepthwiseConv3(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [8, 4, 12, 12]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 5, 5]

    def init_data_format(self):
        self.data_format = "NCHW"


@unittest.skip("tecodnn not support dilation > 1")
class TestDepthwiseConvWithDilation(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]


@unittest.skip("tecodnn not support dilation > 1")
class TestDepthwiseConvWithDilation2(TestDepthwiseConvWithDilation):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]


class TestDepthwiseConvandFuse(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [8, 4, 3, 96]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 5, 5]


class TestDepthwiseConv2andFuse(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [8, 4, 12, 12]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 5, 5]


class TestDepthwiseConv3andFuse(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [8, 4, 3, 96]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 5, 5]


@unittest.skip("tecodnn not support dilation > 1")
class TestDepthwiseConvWithDilationandFuse(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]


@unittest.skip("tecodnn not support dilation > 1")
class TestDepthwiseConvWithDilation2andFuse(TestDepthwiseConvWithDilationandFuse):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConv_AsyPadding(TestDepthwiseConv2DOp):
    def init_test_case(self):
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [1, 1, 0, 1]
        self.padding_algorithm = "EXPLICIT"


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConv2_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [0, 1, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv3_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [8, 4, 3, 96]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 5, 5]

    def init_paddings(self):
        self.pad = [1, 1, 0, 0]
        self.padding_algorithm = "EXPLICIT"


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConvWithDilation_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [1, 1, 2, 1]
        self.padding_algorithm = "EXPLICIT"

    def init_dilation(self):
        self.dilations = [2, 2]


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConvWithDilation2_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [0, 1, 1, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_dilation(self):
        self.dilations = [2, 2]


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConvandFuse_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 2, 3]
        self.padding_algorithm = "EXPLICIT"


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConv2andFuse_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [1, 1, 1, 2]
        self.padding_algorithm = "EXPLICIT"


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConv3andFuse_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [1, 2, 0, 2]
        self.padding_algorithm = "EXPLICIT"


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConvWithDilationandFuse_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"
        self.python_api = depthwise_conv2d_wrapper

    def init_paddings(self):
        self.pad = [2, 1, 1, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_dilation(self):
        self.dilations = [2, 2]


@unittest.skip("AsyPadding not supported")
class TestDepthwiseConvWithDilation2andFuse_AsyPadding(TestDepthwiseConv_AsyPadding):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [1, 3, 1, 3]
        self.padding_algorithm = "EXPLICIT"

    def init_dilation(self):
        self.dilations = [2, 2]


class TestDepthwiseConvOpError(unittest.TestCase):
    def test_errors(self):
        def test_define_error_group_channel():
            paddle.device.set_device("sdaa")
            paddle.disable_static()
            input_size = [4, 128, 128, 6]
            weight_size = [6, 6, 7, 7]
            x = np.random.uniform(low=0, high=1.0, size=input_size)
            w = np.random.uniform(low=0, high=1.0, size=weight_size)
            x_var = paddle.to_tensor(x, dtype="float32")
            w_var = paddle.to_tensor(w, dtype="float32")
            out = _C_ops.depthwise_conv2d(
                x_var,
                w_var,
                [1, 1],
                [0, 0],
                "VALID",
                1,
                [1],
                "NHWC",
                False,
                -1,
                False,
                False,
            )

        paddle.disable_static()
        self.assertRaises(ValueError, test_define_error_group_channel)
        paddle.enable_static()


create_test_padding_SAME_class(TestDepthwiseConv2DOp)
create_test_padding_SAME_class(TestDepthwiseConvWithDilation)
create_test_padding_SAME_class(TestDepthwiseConvandFuse)
create_test_padding_SAME_class(TestDepthwiseConvWithDilationandFuse)

create_test_padding_VALID_class(TestDepthwiseConv2DOp)
create_test_padding_VALID_class(TestDepthwiseConvWithDilation)
create_test_padding_VALID_class(TestDepthwiseConvandFuse)
create_test_padding_VALID_class(TestDepthwiseConvWithDilationandFuse)

# # channel last
create_test_channel_last_class(TestDepthwiseConv2DOp)
create_test_channel_last_class(TestDepthwiseConvWithDilation)
create_test_channel_last_class(TestDepthwiseConvandFuse)
create_test_channel_last_class(TestDepthwiseConvWithDilationandFuse)

create_test_fp16_class(TestDepthwiseConv2DOp)
create_test_fp16_class(TestDepthwiseConvWithDilation)
create_test_fp16_class(TestDepthwiseConvandFuse)
create_test_fp16_class(TestDepthwiseConvWithDilationandFuse)

if __name__ == "__main__":
    unittest.main()
