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


class TestGaussianRandomKernel(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "truncated_gaussian_random"
        self.python_api = paddle._C_ops.truncated_gaussian_random
        self.init_dtype()
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {"Out": np.zeros((123, 92), dtype="float32")}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_output(self):
        self.check_output_customized(self.verify_output, self.place)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        self.assertTrue(
            np.allclose(hist, hist2, rtol=0, atol=0.02),
            "hist: " + str(hist) + " hist2: " + str(hist2),
        )


@unittest.skip("FP16 not supported")
class TestGaussianRandomKernelFP16(OpTest):
    def init_dtype(self):
        self.dtype = np.float16


class TestPrecisionAlignment(unittest.TestCase):
    def test_verify_output_with_GPU(self):
        paddle.seed(2023)
        paddle.disable_static()
        data = paddle.ones(shape=[3, 1, 8], dtype="float32")
        bias_attr = paddle.framework.ParamAttr(
            name="linear_bias",
            initializer=paddle.nn.initializer.TruncatedNormal(mean=1.0, std=2.0),
        )
        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.TruncatedNormal(mean=1.0, std=2.0),
        )
        linear = paddle.nn.Linear(8, 8, weight_attr=weight_attr, bias_attr=bias_attr)
        res = linear(data)

        weight_attr_gpu = paddle.to_tensor(
            [
                [
                    -2.01012564,
                    -2.44575310,
                    1.17401874,
                    2.16073179,
                    -0.76169139,
                    2.23397636,
                    2.56214452,
                    -0.78591233,
                ],
                [
                    3.01231623,
                    -1.13635075,
                    -0.90178585,
                    1.83117115,
                    2.94593811,
                    2.29788685,
                    0.76837879,
                    -0.6240067,
                ],
                [
                    0.66532052,
                    -1.86164320,
                    -1.13086987,
                    0.78322679,
                    -2.12622166,
                    3.37603354,
                    1.91009712,
                    -2.53848076,
                ],
                [
                    2.61308789,
                    -0.71218991,
                    -0.49577177,
                    2.04250932,
                    1.22324121,
                    2.68732953,
                    -2.44257021,
                    2.76428628,
                ],
                [
                    -2.75118279,
                    1.62414622,
                    4.00820446,
                    2.18627882,
                    -1.07511091,
                    -1.76093519,
                    0.71191508,
                    0.58216548,
                ],
                [
                    4.62944603,
                    0.81320274,
                    3.25842428,
                    -0.06121963,
                    -1.47933340,
                    0.84862852,
                    4.30057287,
                    -1.09468853,
                ],
                [
                    0.31158024,
                    1.90019679,
                    3.06459785,
                    2.99715447,
                    1.37522984,
                    1.49921751,
                    1.62003994,
                    0.94212717,
                ],
                [
                    2.46152616,
                    0.73536116,
                    1.90530849,
                    0.68263239,
                    0.45855901,
                    1.55757499,
                    -1.28208399,
                    0.50616789,
                ],
            ]
        )
        bias_attr_gpu = paddle.to_tensor(
            [
                3.01231623,
                -1.13635075,
                -0.90178585,
                1.83117115,
                2.94593811,
                2.29788685,
                0.76837879,
                -0.62400675,
            ]
        )

        bias_sdaa_npy = linear.bias.numpy()
        weight_sdaa_npy = linear.weight.numpy()
        bias_gpu_npy = bias_attr_gpu.numpy()
        weight_gpu_npy = weight_attr_gpu.numpy()

        np.testing.assert_allclose(bias_sdaa_npy, bias_gpu_npy, rtol=1e-05)
        np.testing.assert_allclose(weight_sdaa_npy, weight_gpu_npy, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
