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

import unittest

import numpy as np
import paddle
from paddle import base
from paddle.base import core
from op_test import OpTest

paddle.enable_static()
SEED = 2021


def coalesce_tensor_eager_api(
    Input,
    datatype=core.VarDesc.VarType.FP32,
    copy_data=False,
    set_constant=False,
    persist_output=False,
    constant=0.0,
    use_align=True,
    align_size=-1,
    user_defined_size_of_dtype=-1,
    concated_shapes=[],
    concated_ranks=[],
):
    if datatype == int(core.VarDesc.VarType.FP32):
        datatype = core.VarDesc.VarType.FP32
    return paddle._C_ops.coalesce_tensor(
        Input,
        datatype,
        copy_data,
        set_constant,
        persist_output,
        constant,
        use_align,
        align_size,
        user_defined_size_of_dtype,
        concated_shapes,
        concated_ranks,
    )


class TestAllocContinuousSpace(OpTest):
    def setUp(self):
        self.__class__.use_custom_device = True
        self.op_type = "coalesce_tensor"
        self.python_api = coalesce_tensor_eager_api
        self.dtype, self.base_dtype = self.init_dtype()
        self.attrs = self.init_attr()
        self.copy_data = self.attrs["copy_data"]
        self.constant = self.attrs["constant"]
        self.set_constant = self.attrs["set_constant"]
        self.Inputs = self.init_input()
        self.Outputs, self.FusedOutput = self.init_output(
            self.Inputs, self.set_constant, self.constant
        )
        self.inputs = {"Input": self.Inputs}
        self.outputs = {"Output": self.Outputs, "FusedOutput": self.FusedOutput}

    def init_dtype(self):
        return np.float32, int(core.VarDesc.VarType.FP32)

    def init_input(self):
        inputs = []
        inputs.append(("x1", np.random.random([20, 3]).astype(self.dtype)))
        inputs.append(("x2", np.random.random([20]).astype(self.dtype)))
        inputs.append(("x3", np.random.random([1]).astype(self.dtype)))
        inputs.append(("x4", np.random.random([200, 30]).astype(self.dtype)))
        inputs.append(("x5", np.random.random([30]).astype(self.dtype)))
        inputs.append(("x6", np.random.random([1]).astype(self.dtype)))
        return inputs

    def init_attr(self):
        return {
            "copy_data": True,
            "set_constant": False,
            "constant": 0.0,
            "use_align": True,
            "dtype": self.base_dtype,
        }

    def init_output(self, input_list, set_constant, constant):
        inputs = []
        outputs = input_list
        # SDAAMinChunkSize=8k bytes, FP32=4 bytes
        alignment = 8192 / 4
        if "user_defined_size_of_dtype" in self.attrs:
            alignment = 8192 / self.attrs["user_defined_size_of_dtype"]

        for input in input_list:
            length = len(input[1].flatten())
            aligned_len = (length + alignment) // alignment * alignment
            out = np.zeros(int(aligned_len))
            out[0:length] = input[1].flatten()
            inputs.append(out)

        coalesce_tensor_var = np.concatenate([input for input in inputs])
        if set_constant:
            coalesce_tensor_var = np.ones((len(coalesce_tensor_var))) * constant
            outputs = [
                (out[0], np.ones(out[1].shape).astype(self.dtype) * constant)
                for out in outputs
            ]
        return outputs, coalesce_tensor_var

    def verify_output(self, place):
        with base.dygraph.base.guard(place=place):
            tensor_input = [
                base.dygraph.base.to_variable(value=data[1])
                for data in self.inputs["Input"]
            ]
            eager_outputs, eager_fused_output = coalesce_tensor_eager_api(
                tensor_input,
                datatype=self.attrs["dtype"],
                copy_data=self.attrs["copy_data"]
                if "copy_data" in self.attrs
                else False,
                set_constant=self.attrs["set_constant"]
                if "set_constant" in self.attrs
                else False,
                persist_output=False,
                constant=self.attrs["constant"] if "constant" in self.attrs else 0.0,
                use_align=True,
                align_size=-1,
                user_defined_size_of_dtype=self.attrs["user_defined_size_of_dtype"]
                if "user_defined_size_of_dtype" in self.attrs
                else -1,
                concated_shapes=[],
                concated_ranks=[],
            )
            for idx, (expected, eager_output) in enumerate(
                zip(self.outputs["Output"], eager_outputs)
            ):
                np.testing.assert_allclose(
                    expected[1],
                    eager_output,
                    atol=1e-5,
                    err_msg=f"not equal {idx}",
                )
            np.testing.assert_allclose(
                self.outputs["FusedOutput"],
                eager_fused_output,
                atol=1e-5,
                err_msg="not equal fusedoutput",
            )

    def test_check_output(self):
        self.check_output_with_place(
            place=paddle.CustomPlace("sdaa", 0),
            no_check_set=["FusedOutput"],
            atol=1e-5,
            check_dygraph=False,
        )
        self.verify_output(paddle.CustomPlace("sdaa", 0))


class TestAllocContinuousSpace2(TestAllocContinuousSpace):
    def init_attr(self):
        return {
            "copy_data": False,
            "set_constant": True,
            "constant": 0.5,
            "use_align": True,
            "dtype": self.base_dtype,
            "user_defined_size_of_dtype": 2,
        }

    def test_check_output(self):
        self.check_output_with_place(
            place=paddle.CustomPlace("sdaa", 0),
            no_check_set=["FusedOutput"],
            atol=1e-5,
            check_dygraph=False,
        )
        self.verify_output(paddle.CustomPlace("sdaa", 0))


if __name__ == "__main__":
    unittest.main()
