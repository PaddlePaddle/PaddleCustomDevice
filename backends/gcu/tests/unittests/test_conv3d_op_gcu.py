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

import unittest

import numpy as np
import paddle
import paddle.base as base
from tests.op_test import OpTest

paddle.enable_static()


def conv3d_forward_naive(
    input, filter, group, conv_param, padding_algorithm="EXPLICIT", data_format="NCDHW"
):

    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError(
            "Unknown Attr(padding_algorithm): '%s'. "
            "It can only be 'SAME' or 'VALID'." % str(padding_algorithm)
        )

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Unknown Attr(data_format): '%s' ."
            "It can only be 'NCDHW' or 'NDHWC'." % str(data_format)
        )

    channel_last = data_format == "NDHWC"
    if channel_last:
        input = np.transpose(input, [0, 4, 1, 2, 3])

    in_n, in_c, in_d, in_h, in_w = input.shape

    f_n, f_c, f_d, f_h, f_w = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group

    stride, pad, dilation = (
        conv_param["stride"],
        conv_param["pad"],
        conv_param["dilations"],
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

    ksize = filter.shape[2:5]
    if padding_algorithm == "VALID":
        pad = [0, 0, 0, 0, 0, 0]
    elif padding_algorithm == "SAME":
        dilation = [1, 1, 1]
        input_data_shape = input.shape[2:5]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_d_0, pad_d_1 = pad[0], pad[0]
    pad_h_0, pad_h_1 = pad[1], pad[1]
    pad_w_0, pad_w_1 = pad[2], pad[2]
    if len(pad) == 6:
        pad_d_0, pad_d_1 = pad[0], pad[1]
        pad_h_0, pad_h_1 = pad[2], pad[3]
        pad_w_0, pad_w_1 = pad[4], pad[5]

    out_d = 1 + (in_d + pad_d_0 + pad_d_1 - (dilation[0] * (f_d - 1) + 1)) // stride[0]
    out_h = 1 + (in_h + pad_h_0 + pad_h_1 - (dilation[1] * (f_h - 1) + 1)) // stride[1]
    out_w = 1 + (in_w + pad_w_0 + pad_w_1 - (dilation[2] * (f_w - 1) + 1)) // stride[2]

    out = np.zeros((in_n, out_c, out_d, out_h, out_w))

    d_bolck_d = dilation[0] * (f_d - 1) + 1
    d_bolck_h = dilation[1] * (f_h - 1) + 1
    d_bolck_w = dilation[2] * (f_w - 1) + 1

    input_pad = np.pad(
        input,
        ((0, 0), (0, 0), (pad_d_0, pad_d_1), (pad_h_0, pad_h_1), (pad_w_0, pad_w_1)),
        mode="constant",
        constant_values=0,
    )

    filter_dilation = np.zeros((f_n, f_c, d_bolck_d, d_bolck_h, d_bolck_w))
    filter_dilation[
        :,
        :,
        0 : d_bolck_d : dilation[0],
        0 : d_bolck_h : dilation[1],
        0 : d_bolck_w : dilation[2],
    ] = filter

    for d in range(out_d):
        for i in range(out_h):
            for j in range(out_w):
                for g in range(group):
                    input_pad_masked = input_pad[
                        :,
                        g * f_c : (g + 1) * f_c,
                        d * stride[0] : d * stride[0] + d_bolck_d,
                        i * stride[1] : i * stride[1] + d_bolck_h,
                        j * stride[2] : j * stride[2] + d_bolck_w,
                    ]

                    f_sub = filter_dilation[g * sub_f_n : (g + 1) * sub_f_n, :, :, :, :]
                    for k in range(sub_out_c):
                        out[:, g * sub_out_c + k, d, i, j] = np.sum(
                            input_pad_masked * f_sub[k, :, :, :, :], axis=(1, 2, 3, 4)
                        )
    if channel_last:
        out = np.transpose(out, [0, 2, 3, 4, 1])
    return out


def create_test_fp16_class(parent):
    class TestFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


class TestConv3DOp(OpTest):
    def setUp(self):
        self.op_type = "conv3d"
        self.set_device()
        self.init_dtype()
        self.init_data_format()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv3d_param = {
            "stride": self.stride,
            "pad": self.pad,
            "dilations": self.dilations,
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)
        output = conv3d_forward_naive(
            input,
            filter,
            self.groups,
            conv3d_param,
        ).astype(self.dtype)

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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = base.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(
            self.place,
            {"Input", "Filter"},
            "Output",
            max_relative_error=0.03,
            numeric_place=paddle.CPUPlace(),
        )

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            return

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
            return

        self.check_grad_with_place(
            self.place,
            ["Filter"],
            "Output",
            max_relative_error=0.03,
            no_grad_set=set(["Input"]),
            numeric_place=paddle.CPUPlace(),
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_data_format(self):
        self.data_format = "NCDHW"

    def init_group(self):
        self.groups = 1

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


class TestCase1(TestConv3DOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


# ---- test asymmetric padding ----
class TestConv3DOp_2(OpTest):
    def setUp(self):
        self.op_type = "conv3d"
        self.set_device()
        self.init_dtype()
        self.init_data_format()
        self.init_group()
        self.init_dilation()
        self.init_paddings()
        self.init_test_case()

        self.init_test_case_2()

        conv3d_param = {
            "stride": self.stride,
            "pad": self.pad,
            "dilations": self.dilations,
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)
        output = conv3d_forward_naive(
            input,
            filter,
            self.groups,
            conv3d_param,
            self.padding_algorithm,
            self.data_format,
        ).astype(self.dtype)

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

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = base.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(paddle.CustomPlace("gcu", 0), atol=1e-2)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(
            self.place,
            {"Input", "Filter"},
            "Output",
            max_relative_error=0.03,
            numeric_place=paddle.CPUPlace(),
        )

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            return

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
            return

        self.check_grad_with_place(
            self.place,
            ["Filter"],
            "Output",
            max_relative_error=0.03,
            no_grad_set=set(["Input"]),
            numeric_place=paddle.CPUPlace(),
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_data_format(self):
        self.data_format = "NCDHW"

    def init_group(self):
        self.groups = 1

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_paddings(self):
        self.pad = [0, 0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_test_case_2(self):
        pass


class TestConv3DOp_AsyPadding(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_paddings(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestConv3DOp_DiffDataInDiffDim(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 5, 5]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 4, 3]

    def init_paddings(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestCase1_AsyPadding(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_paddings(self):
        self.pad = [0, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


# --------- test python API ---------------
class TestConv3DAPI(unittest.TestCase):
    def test_api(self):

        input_NDHWC = paddle.static.data(
            name="input_NDHWC",
            shape=[2, 5, 5, 5, 3],
            dtype="float32",
        )

        input_NCDHW = paddle.static.data(
            name="input_NCDHW",
            shape=[2, 3, 5, 5, 3],
            dtype="float32",
        )

        paddle.static.nn.conv3d(
            input=input_NDHWC,
            num_filters=3,
            filter_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=0,
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )

        paddle.static.nn.conv3d(
            input=input_NCDHW,
            num_filters=3,
            filter_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[1, 2, 1, 0, 1, 0],
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )

        paddle.static.nn.conv3d(
            input=input_NCDHW,
            num_filters=3,
            filter_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]],
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )

        paddle.static.nn.conv3d(
            input=input_NDHWC,
            num_filters=3,
            filter_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
            dilation=[1, 1, 1],
            groups=1,
            data_format="NDHWC",
        )

        paddle.static.nn.conv3d(
            input=input_NCDHW,
            num_filters=3,
            filter_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding="SAME",
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )

        paddle.static.nn.conv3d(
            input=input_NCDHW,
            num_filters=3,
            filter_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding="VALID",
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )


class TestConv3DAPI_Error(unittest.TestCase):
    def test_api(self):
        input = paddle.static.data(
            name="input",
            shape=[2, 5, 5, 5, 4],
            dtype="float32",
        )

        # ValueError: cudnn
        def run_1():
            paddle.static.nn.conv3d(
                input=input,
                num_filters=3,
                filter_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                use_cudnn=[0],
                data_format="NCDHW",
            )

        self.assertRaises(ValueError, run_1)

        # ValueError: data_format
        def run_2():
            paddle.static.nn.conv3d(
                input=input,
                num_filters=3,
                filter_size=[3, 3, 3],
                stride=[1, 1, 1],
                padding=0,
                dilation=[1, 1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NCHWC",
            )

        self.assertRaises(ValueError, run_2)

        # ValueError: padding
        def run_3():
            paddle.static.nn.conv3d(
                input=input,
                num_filters=3,
                filter_size=3,
                stride=1,
                padding="SAMEE",
                dilation=1,
                groups=1,
                use_cudnn=False,
                data_format="NCDHW",
            )

        self.assertRaises(ValueError, run_3)

        def run_4():
            paddle.static.nn.conv3d(
                input=input,
                num_filters=3,
                filter_size=3,
                stride=1,
                padding=[[0, 1], [0, 0], [0, 1], [0, 1], [0, 1]],
                dilation=1,
                groups=1,
                use_cudnn=False,
                data_format="NCDHW",
            )

        self.assertRaises(ValueError, run_4)

        def run_5():
            paddle.static.nn.conv3d(
                input=input,
                num_filters=3,
                filter_size=0,
                stride=0,
                padding=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                dilation=1,
                groups=1,
                use_cudnn=False,
                data_format="NDHWC",
            )

        self.assertRaises(ValueError, run_5)

        # ValueError: channel dimmention
        x = paddle.static.data(name="x", shape=[2, 5, 5, 5, -1], dtype="float32")

        def run_6():
            paddle.static.nn.conv3d(
                input=x,
                num_filters=3,
                filter_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                use_cudnn=False,
                data_format="NDHWC",
            )

        self.assertRaises(ValueError, run_6)

        # ValueError: groups
        def run_7():
            paddle.static.nn.conv3d(
                input=input,
                num_filters=3,
                filter_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=3,
                use_cudnn=False,
                data_format="NDHWC",
            )

        self.assertRaises(ValueError, run_7)

        # ValueError: filter num
        def run_8():
            paddle.static.nn.conv3d(
                input=input,
                num_filters=0,
                filter_size=0,
                stride=0,
                padding=0,
                dilation=0,
                groups=1,
                use_cudnn=False,
                data_format="NDHWC",
            )

        self.assertRaises(ValueError, run_8)


create_test_fp16_class(TestConv3DOp)
create_test_fp16_class(TestConv3DOp_2)


if __name__ == "__main__":
    unittest.main()
