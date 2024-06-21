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
import paddle

paddle.enable_static()

SEED = 2021
EPOCH = 100


# @skip_check_grad_ci("haven't implemention")
class TestSliceOp(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()
        self.inputs = {"Input": self.input}
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def init_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Out",
                max_relative_error=0.02,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Out",
            )

    def check_grad_with_place(
        self,
        place,
        inputs_to_check,
        output_names,
        no_grad_set=None,
        numeric_grad_delta=0.005,
        in_place=False,
        max_relative_error=0.005,
        user_defined_grads=None,
        user_defined_grad_outputs=None,
        check_dygraph=True,
        numeric_place=None,
    ):
        if self.dtype == np.float32:
            numeric_place = paddle.CPUPlace()

        super().check_grad_with_place(
            place,
            inputs_to_check,
            output_names,
            no_grad_set,
            numeric_grad_delta,
            in_place,
            max_relative_error,
            user_defined_grads,
            user_defined_grad_outputs,
            check_dygraph,
            numeric_place=numeric_place,
        )


class TestSliceMultiDim(TestSliceOp):
    def config(self):
        self.input = np.random.random([3, 2, 5, 4, 8]).astype(self.dtype)
        self.starts = [0]
        self.ends = [1]
        self.axes = [4]
        self.infer_flags = [1]
        self.out = self.input[:, :, :, :, 0:1]


class TestSliceOp2(TestSliceOp):
    def config(self):
        self.input = np.random.random([10, 5, 6]).astype(self.dtype)
        self.starts = [0]
        self.ends = [1]
        self.axes = [1]
        self.infer_flags = [1]
        self.out = self.input[:, 0:1, :]


class TestSliceOpFp16(TestSliceOp):
    def init_dtype(self):
        self.dtype = np.float16

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestSliceOpDouble(TestSliceOp):
    def init_dtype(self):
        self.dtype = np.double

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestSliceOpTensor(TestSliceOp):
    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()
        self.inputs = {
            "Input": self.input,
            "StartsTensor": self.starts,
            "EndsTensor": self.ends,
        }
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": [-1, -1, -1],
            "ends": [-1, -1, -1],
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = np.array([1, 0, 2]).astype("int32")
        self.ends = np.array([3, 3, 4]).astype("int32")
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]


class TestSliceOpTensor2(TestSliceOpTensor):
    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()
        self.inputs = {
            "Input": self.input,
            "StartsTensor": self.starts,
            "EndsTensor": self.ends,
        }
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": [-1],
            "ends": [-1],
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([10, 5, 6]).astype(self.dtype)
        self.starts = np.array([0]).astype("int32")
        self.ends = np.array([1]).astype("int32")
        self.axes = [1]
        self.infer_flags = [-1]
        self.out = self.input[:, 0:1, :]


class TestSliceOpFp16Tensor(TestSliceOpTensor):
    def init_dtype(self):
        self.dtype = np.float16

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestSliceOpDoubleTensor(TestSliceOpTensor):
    def init_dtype(self):
        self.dtype = np.double

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestSliceOpTensorList(TestSliceOp):
    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()

        self.starts_tensor_list = []
        for index, ele in enumerate(self.starts):
            self.starts_tensor_list.append(
                ("start" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.ends_tensor_list = []
        for index, ele in enumerate(self.ends):
            self.ends_tensor_list.append(
                ("end" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.inputs = {
            "Input": self.input,
            "StartsTensorList": self.starts_tensor_list,
            "EndsTensorList": self.ends_tensor_list,
        }
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": [-1, -1, -1],
            "ends": [-1, -1, -1],
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]


class TestSliceOpTensorList2(TestSliceOpTensorList):
    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()

        self.starts_tensor_list = []
        for index, ele in enumerate(self.starts):
            self.starts_tensor_list.append(
                ("start" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.ends_tensor_list = []
        for index, ele in enumerate(self.ends):
            self.ends_tensor_list.append(
                ("end" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.inputs = {
            "Input": self.input,
            "StartsTensorList": self.starts_tensor_list,
            "EndsTensorList": self.ends_tensor_list,
        }
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": [-1],
            "ends": [-1],
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([10, 5, 6]).astype(self.dtype)
        self.starts = np.array([0]).astype("int32")
        self.ends = np.array([1]).astype("int32")
        self.axes = [1]
        self.infer_flags = [-1]
        self.out = self.input[:, 0:1, :]


class TestSliceOpFp16TensorList(TestSliceOpTensorList):
    def init_dtype(self):
        self.dtype = np.float16

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)


# @skip_check_grad_ci("not")
class TestSliceOpDecsDim(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()
        self.set_inputs()
        self.set_outputs()
        self.set_attrs()

    def set_inputs(self):
        self.inputs = {"Input": self.input}

    def set_outputs(self):
        self.outputs = {"Out": self.out}

    def set_attrs(self):
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
            "decrease_axis": self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

    def init_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Out",
                max_relative_error=0.5,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["Input"],
                "Out",
            )


class TestSliceOpDecsDimFp16(TestSliceOpDecsDim):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpDecsDim2(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0, 2:4, :]


class TestSliceOpDecsDim3(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [-1, 0, 2]
        self.ends = [1000000, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-1, 0, 2:4, :]


class TestSliceOpDecsDim4(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 7]).astype(self.dtype)
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4][0]


class TestSliceOpDecsDim5(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [3]
        self.decrease_axis = [3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[:, :, :, -1]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSliceOpDecsDim6(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4][0]


class TestSliceOpDecsDimStartsTensor(TestSliceOpDecsDim):
    def set_inputs(self):
        self.inputs = {
            "Input": self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
        }

    def set_attrs(self):
        self.attrs = {
            "axes": self.axes,
            # 'starts': self.starts,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
            "decrease_axis": self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0:3, 2:4, :]


class TestSliceOpDecsDimStartsTensorFP16(TestSliceOpDecsDimStartsTensor):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpDecsDimStartsTensorStartsAndEndsTensor(TestSliceOpDecsDim):
    def set_inputs(self):
        self.inputs = {
            "Input": self.input,
            "StartsTensor": np.array(self.starts, dtype="int64"),
            "EndsTensor": np.array(self.ends, dtype="int32"),
        }

    def set_attrs(self):
        self.attrs = {
            "axes": self.axes,
            # 'starts': self.starts,
            # 'ends': self.ends,
            "infer_flags": self.infer_flags,
            "decrease_axis": self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0, 2:4, :]


class TestSliceOpDecsDimStartsTensorStartsAndEndsTensorFP16(
    TestSliceOpDecsDimStartsTensorStartsAndEndsTensor
):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpDecsDimStartsListTensor(TestSliceOpDecsDim):
    def set_inputs(self):
        starts_tensor = []
        for index, ele in enumerate(self.starts):
            starts_tensor.append(("x" + str(index), np.ones((1)).astype("int32") * ele))

        self.inputs = {"Input": self.input, "StartsTensorList": starts_tensor}

    def set_attrs(self):
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts_infer,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
            "decrease_axis": self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, -1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

        self.starts_infer = [1, -1, 2]


class TestSliceOpDecsDimStartsListTensor2(TestSliceOpDecsDimStartsListTensor):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [3]
        self.decrease_axis = [3]
        self.infer_flags = [-1]
        self.out = self.input[:, :, :, -1]

        self.starts_infer = [-1]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSliceOpDecsDimStartsListTensorFP16(TestSliceOpDecsDimStartsListTensor):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpInt64(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()
        self.inputs = {"Input": self.input}
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.randint(100, size=(3, 4, 5, 6)).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def init_dtype(self):
        self.dtype = np.int64

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSliceOpTensorInt64(TestSliceOpInt64):
    def setUp(self):
        self.op_type = "slice"
        self.python_api = paddle.slice
        self.set_sdaa()
        self.init_dtype()
        self.config()
        self.inputs = {
            "Input": self.input,
            "StartsTensor": self.starts,
            "EndsTensor": self.ends,
        }
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": [-1, -1, -1],
            "ends": [-1, -1, -1],
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.randint(100, size=(3, 4, 5, 6)).astype(self.dtype)
        self.starts = np.array([1, 0, 2]).astype("int32")
        self.ends = np.array([3, 3, 4]).astype("int32")
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSliceOpBool(TestSliceOpInt64):
    def init_dtype(self):
        self.dtype = np.bool_


class TestSliceOpTensorBool(TestSliceOpTensorInt64):
    def init_dtype(self):
        self.dtype = np.bool_


class TestSliceOpIndexApi(OpTest):
    def setUp(self):
        self.python_api = paddle.slice
        self.__class__.no_need_check_grad = False
        self.__class__.op_type = "slice"
        self.set_sdaa()
        self.init_dtype()

    def init_dtype(self):
        self.dtype = "float32"

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_api_dygraph(self):
        paddle.disable_static()
        paddle.set_device("sdaa")
        input = paddle.to_tensor([0.0], dtype=self.dtype)
        output_sdaa = input[0]
        assert "sdaa" in str(output_sdaa.place)
        paddle.set_device("cpu")
        input = paddle.to_tensor([0.0], dtype=self.dtype)
        output_cpu = input[0]
        assert "cpu" in str(output_cpu.place)
        np.testing.assert_allclose(output_cpu.numpy(), output_sdaa.numpy(), atol=1e-10)


class TestSliceOpIndexApi_double(TestSliceOpIndexApi):
    def init_dtype(self):
        self.dtype = "float64"


if __name__ == "__main__":
    unittest.main()
