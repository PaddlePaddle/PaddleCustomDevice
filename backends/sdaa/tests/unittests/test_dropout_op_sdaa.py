#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from unittest import mock
from op_test import OpTest, skip_check_grad_ci
import os
import paddle
import paddle.base as base
import random
from typing import List

paddle.enable_static()

SEED = 2021


def dropout_wapper(
    X,
    Seed=None,
    dropout_prob=0.5,
    is_test=False,
    dropout_implementation="downgrade_in_infer",
    seed=0,
    fix_seed=False,
):
    return paddle._C_ops.dropout(
        X,
        Seed,
        dropout_prob,
        is_test,
        dropout_implementation,
        seed,
        fix_seed,
    )


class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wapper
        self.set_sdaa()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": np.ones((32, 64)).astype("uint8"),
        }

    def init_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            pass
        else:
            self.check_grad_with_place(
                self.place,
                ["X"],
                "Out",
                numeric_place=paddle.CPUPlace(),
            )


class TestDropoutOpInput1d(TestDropoutOp):
    # change input shape
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wapper
        self.set_sdaa()
        self.init_dtype()
        self.inputs = {"X": np.random.random((3, 62)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": np.ones((3, 62)).astype("uint8"),
        }


class TestDropoutOpInput1d_1(TestDropoutOp):
    # the input is 1-D
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wapper
        self.set_sdaa()
        self.init_dtype()
        self.inputs = {"X": np.random.random((2000)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": np.ones((2000)).astype("uint8"),
        }


class TestDropoutOp2(TestDropoutOp):
    # the dropout_prob is 1.0
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wapper
        self.set_sdaa()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 1.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": np.zeros((32, 64)).astype("float32"),
            "Mask": np.zeros((32, 64)).astype("uint8"),
        }


class TestDropoutOp3(TestDropoutOp):
    # the input dim is 3
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wapper
        self.set_sdaa()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64, 2)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": np.ones((32, 64, 2)).astype("uint8"),
        }


@skip_check_grad_ci(reason="is_test is for inference, check_grad is not needed")
class TestDropoutOpWithIsTest(TestDropoutOp):
    # the seed is a Tensor
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wapper
        self.set_sdaa()
        self.init_dtype()
        self.inputs = {
            "X": np.random.random((32, 64)).astype(self.dtype),
            "Seed": np.asarray([125], dtype="int32"),
        }
        self.attrs = {
            "dropout_prob": 0.0,
            "is_test": True,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
        }

    def test_check_grad_normal(self):
        pass


class TestDropoutOpWithSeed(TestDropoutOp):
    # the seed is a Tensor
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wapper
        self.set_sdaa()
        self.init_dtype()
        self.inputs = {
            "X": np.random.random((32, 64)).astype(self.dtype),
            "Seed": np.asarray([125], dtype="int32"),
        }
        self.attrs = {
            "dropout_prob": 0.0,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": np.ones((32, 64)).astype("uint8"),
        }


class TestDropoutOpFp16(TestDropoutOp):
    # float16
    def init_dtype(self):
        self.dtype = np.float16

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        # self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestDropoutAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [base.CPUPlace(), paddle.CustomPlace("sdaa", 0)]

    def check_static_result(self, place):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(name="input", shape=[40, 40], dtype="float32")
            res1 = paddle.nn.functional.dropout(
                x=input, p=0.0, training=False, mode="upscale_in_train"
            )
            res2 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=True, mode="upscale_in_train"
            )
            res3 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=False, mode="upscale_in_train"
            )
            res4 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=[0, 1], training=True, mode="upscale_in_train"
            )
            res5 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=[0, 1], training=False, mode="upscale_in_train"
            )
            res6 = paddle.nn.functional.dropout(
                x=input, p=1.0, training=True, mode="upscale_in_train"
            )
            res7 = paddle.nn.functional.dropout(x=input, p=0.0, mode="upscale_in_train")
            res8 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=(0, 1), training=False, mode="upscale_in_train"
            )

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np2 = np.zeros_like(in_np)

            exe = base.Executor(place)
            res_list = [res1, res2, res3, res4, res5, res7, res8]
            for res in res_list:
                fetches = exe.run(
                    base.default_main_program(),
                    feed={"input": in_np},
                    fetch_list=[res],
                )
                self.assertTrue(np.allclose(fetches[0], res_np))
            fetches2 = exe.run(
                base.default_main_program(), feed={"input": in_np}, fetch_list=[res6]
            )
            self.assertTrue(np.allclose(fetches2[0], res_np2))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class TestDropoutBase(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.init_shape()
        self.init_p()
        self.place = [paddle.CustomPlace("sdaa", 0)]
        self.iter = 10

    def init_dtype(self):
        self.dtype = "float32"

    def init_shape(self):
        self.shape = [2, 3, 3]

    def init_p(self):
        self.p = 0.5

    def _check_abnormal_output(self, x: np.ndarray):
        y, mask = paddle._C_ops.dropout(
            paddle.to_tensor(x), None, self.p, False, "upscale_in_train", 0, False
        )
        y, mask = y.numpy(), mask.numpy()
        np.testing.assert_array_equal(np.where(y > 0, 1, 0), mask)

        # upscale in train
        p = (x * mask) / (1.0 - self.p)
        np.testing.assert_allclose(y, p, rtol=1e-6)

        return y, mask

    def check_output(self, place):
        with paddle.base.dygraph.guard(place=place):
            set_seed(SEED)
            x = np.random.randint(low=1, high=12341234234, size=self.shape).astype(
                self.dtype
            )
            self._check_abnormal_output(x)

    def _check_stability_helper(self, x: List[List[np.ndarray]], abs_eps: float = 1e-8):
        for i in range(1, len(x)):
            for j in range(len(x[i])):
                np.testing.assert_allclose(x[i][j], x[i - 1][j], atol=abs_eps)

    def check_stability(self, place):
        paddle.disable_static(place)
        dropout_res = []
        for j in range(3):
            step_result = []
            set_seed(SEED)
            for i in range(self.iter):
                x = np.random.randint(low=1, high=12341234234, size=self.shape).astype(
                    self.dtype
                )
                y, _ = self._check_abnormal_output(x)
                step_result.append(y)
            dropout_res.append(step_result)

        self._check_stability_helper(dropout_res)

        paddle.enable_static()

    def test_dygraph_api(self):
        for place in self.place:
            self.check_output(place)
            self.check_stability(place)


class TestDropoutBase2(TestDropoutBase):
    def init_p(self):
        self.p = 0.7


class TestDropoutBase3(TestDropoutBase):
    def init_p(self):
        self.p = 0.9


class TestDropoutBaseWithBigShape(TestDropoutBase):
    def init_shape(self):
        self.shape = [512, 3, 512]


def check_bernoulli_distribution(matrix, p, significance_level=0.05):
    shape = matrix.shape
    dtype = matrix.dtype

    proportion = paddle.mean(matrix)

    from scipy import stats

    data = matrix.flatten()
    _, p_value = stats.ks_2samp(data, stats.bernoulli.rvs(1 - p, size=len(data)))
    is_bernoulli = p_value > significance_level

    return is_bernoulli, proportion, shape, dtype


class TestDropoutDistribution(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.init_shape()
        self.init_p()
        self.proportion_eps = 0.2
        self.place = [paddle.CustomPlace("sdaa", 0)]

    def init_dtype(self):
        self.dtype = "float32"

    def init_shape(self):
        self.shape = [4, 3, 8]

    def init_p(self):
        self.p = 0.5

    def check_distribution(self, place):
        paddle.disable_static(place)
        set_seed(SEED)
        x = paddle.to_tensor(np.random.rand(*tuple(self.shape)).astype(self.dtype))
        _, mask = paddle._C_ops.dropout(
            x, None, self.p, False, "upscale_in_train", 0, False
        )
        is_bernoulli, proportion, shape, dtype = check_bernoulli_distribution(
            mask.astype("float32"), p=self.p
        )

        print("Is Bernoulli distribution:", is_bernoulli)
        print("Proportion of 1s:", proportion)
        print("Matrix shape:", shape)
        print("Matrix dtype:", dtype)
        self.assertTrue(is_bernoulli)
        np.testing.assert_allclose(
            proportion.numpy(), 1 - self.p, atol=self.proportion_eps
        )
        paddle.enable_static()

    def test_dygraph_api(self):
        for place in self.place:
            self.check_distribution(place)


class TestDropoutDistribution2(TestDropoutDistribution):
    def init_p(self):
        self.p = 0.7


class TestDropoutDistribution3(TestDropoutDistribution):
    def init_p(self):
        self.p = 0.9


class TestDropoutDistributionWithBigShape(TestDropoutDistribution):
    def init_shape(self):
        self.shape = [8, 512, 768]


class TestDropoutDistributionWithBigShape1(TestDropoutDistribution):
    def init_shape(self):
        self.shape = [8, 512, 768]
        paddle.disable_static()
        out_var = paddle._C_ops.truncated_gaussian_random(
            [100],
            0,
            1,
            0,
            paddle.float32,
            paddle.CustomPlace("sdaa", 0),
        )
        paddle.enable_static()


class TestEnvError(unittest.TestCase):
    @mock.patch.dict(os.environ, {"RANDOM_ALIGN_NV_DEVICE": "123"})
    def test_error(self):
        paddle.disable_static()
        x = paddle.ones([4, 5, 8], dtype=paddle.float32)
        self.assertRaises(ValueError, paddle.nn.functional.dropout, x)
        paddle.enable_static()


class TestAlignGPU(unittest.TestCase):
    def setUp(self):
        paddle.seed(12345)

    @mock.patch.dict(os.environ, {"RANDOM_ALIGN_NV_DEVICE": "v100"})
    def test_error(self):
        paddle.disable_static()

        x = paddle.ones([2, 3, 8], dtype=paddle.float16)
        _, mask = paddle._C_ops.dropout(
            x, None, 0.5, False, "upscale_in_train", 0, False
        )
        mask_gpu = np.array(
            [
                [
                    [0, 1, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [1, 0, 1, 0, 0, 1, 0, 0],
                ],
                [
                    [1, 1, 1, 1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                ],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_equal(mask.numpy(), mask_gpu)

        seed_tensor = paddle.to_tensor([12345], dtype=paddle.int32)
        _, mask_seed_tensor = paddle._C_ops.dropout(
            x, seed_tensor, 0.7, False, "upscale_in_train", 0, True
        )
        _, mask_fix_seed = paddle._C_ops.dropout(
            x, None, 0.7, False, "upscale_in_train", 12345, True
        )
        _, mask = paddle._C_ops.dropout(
            x, None, 0.7, False, "upscale_in_train", 0, False
        )
        mask_gpu = np.array(
            [
                [
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 0, 0, 1, 0],
                ],
                [
                    [0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                ],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_equal(mask_fix_seed.numpy(), mask_gpu)
        np.testing.assert_equal(mask_seed_tensor.numpy(), mask_gpu)
        np.testing.assert_equal(mask.numpy(), mask_gpu)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
