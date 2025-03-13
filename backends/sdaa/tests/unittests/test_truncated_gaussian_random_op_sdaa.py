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
from scipy.stats import truncnorm

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
            "a": self.a,
            "b": self.b,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {"Out": np.zeros((123, 92), dtype="float32")}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0
        self.a = -3.0
        self.b = 5.0

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
        # Note: result of scipy.stats.truncnorm.rvs is inside [loc + a * scale, loc + b * scale], while result of paddle._C_ops.truncated_gaussian_random is inside [a, b]
        data = truncnorm.rvs(a=-2, b=2, loc=1, scale=2, size=(123, 92))
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
        paddle.disable_static()
        paddle.seed(2023)
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
                    -1.6186513,
                    -1.8141448,
                    0.5027355,
                    1.11318,
                    -0.8634622,
                    1.1533377,
                    1.3226961,
                    -0.88003653,
                ],
                [
                    1.5248024,
                    -1.1134394,
                    -0.9586074,
                    0.92273754,
                    1.497296,
                    1.1876942,
                    0.22377466,
                    -0.7683639,
                ],
                [
                    0.15126379,
                    -1.5425761,
                    -1.109894,
                    0.23417574,
                    -1.6748872,
                    1.6611208,
                    0.9697355,
                    -1.8501672,
                ],
                [
                    1.3473688,
                    -0.8294384,
                    -0.67861795,
                    1.0466517,
                    0.53572905,
                    1.3825172,
                    -1.8128731,
                    1.4179268,
                ],
                [
                    -1.9252577,
                    0.79564375,
                    1.8412019,
                    1.1272806,
                    -1.0736177,
                    -1.4884466,
                    0.1841138,
                    0.09239072,
                ],
                [
                    1.9555091,
                    0.25513613,
                    1.619711,
                    -0.3689488,
                    -1.3271068,
                    0.2798395,
                    1.9019394,
                    -1.0863973,
                ],
                [
                    -0.10086287,
                    0.96388614,
                    1.545898,
                    1.5185906,
                    0.6362051,
                    0.71646774,
                    0.7930698,
                    0.3446629,
                ],
                [
                    1.2726623,
                    0.20060308,
                    0.9669079,
                    0.16348132,
                    0.0043795,
                    0.75367326,
                    -1.2062784,
                    0.03834143,
                ],
            ]
        )
        bias_attr_gpu = paddle.to_tensor(
            [
                1.5248024,
                -1.1134394,
                -0.9586074,
                0.92273754,
                1.497296,
                1.1876942,
                0.22377466,
                -0.7683639,
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
