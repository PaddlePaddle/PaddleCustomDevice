# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import os

from op_test import OpTest
from unittest import mock
import paddle

paddle.enable_static()


class TestGaussianRandomKernel(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "gaussian_random"
        self.python_api = paddle.normal
        self.init_dtype()
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "dtype": int(paddle.float32),
        }
        paddle.seed(10)

        self.outputs = {"Out": np.zeros((123, 92), dtype=self.dtype)}

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
        hist = hist.astype(self.dtype)
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype(self.dtype)
        hist2 /= float(outs[0].size)
        self.assertTrue(
            np.allclose(hist, hist2, rtol=0, atol=0.02),
            "hist: " + str(hist) + " hist2: " + str(hist2),
        )


class TestGaussianRandomKernelFP16(TestGaussianRandomKernel):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "gaussian_random"
        self.python_api = paddle.normal
        self.init_dtype()
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "dtype": int(paddle.float16),
        }
        paddle.seed(10)

        self.outputs = {"Out": np.zeros((123, 92), dtype=self.dtype)}

    def init_dtype(self):
        self.dtype = "float16"


class TestGaussianAssertError(unittest.TestCase):
    def setUp(self):
        import os

        paddle.seed(10)
        os.environ["RANDOM_ALIGN_NV_DEVICE"] = "123"

    def test_assert(self):
        paddle.disable_static()
        x = paddle.ones([10, 10], dtype=paddle.float32)
        self.assertRaises(ValueError, paddle.normal, 0, 1, x.shape)
        paddle.enable_static()


class TestGaussianGPUAlign(unittest.TestCase):
    def setUp(self):
        pass

        paddle.seed(10)

    @mock.patch.dict(os.environ, {"RANDOM_ALIGN_NV_DEVICE": "v100"})
    def test_output(self):
        paddle.disable_static(place=paddle.CustomPlace("sdaa", 0))
        result_gpu = np.array(
            [
                [
                    0.79428613,
                    4.36199284,
                    0.45837730,
                    -0.08644140,
                    0.29575956,
                    1.68268943,
                    -2.29122233,
                    4.13956261,
                    0.50201797,
                    0.18817151,
                ],
                [
                    -1.64633179,
                    0.66590738,
                    0.55507219,
                    1.28587222,
                    0.77771586,
                    3.46330786,
                    3.02744174,
                    -0.73102856,
                    0.08613557,
                    1.05707932,
                ],
                [
                    3.53130889,
                    3.70154953,
                    3.45286345,
                    1.55347097,
                    -0.48082912,
                    -0.00646138,
                    -0.22777963,
                    3.33809161,
                    -0.93061662,
                    1.22643197,
                ],
                [
                    -0.85253131,
                    5.01171875,
                    1.70897579,
                    1.34465873,
                    1.12336206,
                    3.81262302,
                    0.56521791,
                    3.66354299,
                    4.12094021,
                    0.12433201,
                ],
                [
                    0.21510941,
                    0.68334508,
                    -0.53453696,
                    -1.32601738,
                    1.39251387,
                    4.61198854,
                    2.45377135,
                    2.56552291,
                    1.44117439,
                    -0.58472776,
                ],
                [
                    2.95232153,
                    0.88248879,
                    1.09343123,
                    -0.39596546,
                    2.24995279,
                    1.90224802,
                    5.89777565,
                    0.62254179,
                    0.87590528,
                    1.15229356,
                ],
                [
                    -2.59120917,
                    2.90204287,
                    3.64452934,
                    1.61680913,
                    -0.41277981,
                    1.45466590,
                    1.33612227,
                    -0.25927031,
                    1.03876650,
                    1.49039960,
                ],
                [
                    -0.74306178,
                    3.03261065,
                    2.29714537,
                    2.30110788,
                    2.56746578,
                    -1.04610276,
                    3.08724189,
                    -0.61205554,
                    1.89334679,
                    0.16544753,
                ],
                [
                    0.87877655,
                    3.77752376,
                    1.77106929,
                    -0.58296728,
                    -1.25932288,
                    1.81657946,
                    3.29656076,
                    3.68044615,
                    1.62799072,
                    2.38805056,
                ],
                [
                    1.43763506,
                    4.98297691,
                    2.89738727,
                    0.93116862,
                    2.89258933,
                    1.89986682,
                    0.74989510,
                    2.70081377,
                    0.70753825,
                    1.99081242,
                ],
            ]
        )
        x = paddle.normal(mean=1.0, std=2.0, shape=[10, 10])
        np.testing.assert_allclose(x.numpy(), result_gpu, atol=1e-6, rtol=1e-4)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
