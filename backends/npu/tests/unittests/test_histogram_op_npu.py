#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from tests.op_test import OpTest

import paddle
from paddle import fluid


class TestHistogramOpError(unittest.TestCase):
    """Test histogram op error."""

    def run_network(self, net_func):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            net_func()
            exe = fluid.Executor()
            exe.run(main_program)

    def test_bins_error(self):
        """Test bins should be greater than or equal to 1."""

        def net_func():
            input_value = paddle.tensor.fill_constant(
                shape=[3, 4], dtype="float32", value=3.0
            )
            paddle.histogram(input=input_value, bins=-1, min=1, max=5)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_min_max_error(self):
        """Test max must be larger or equal to min."""

        def net_func():
            input_value = paddle.tensor.fill_constant(
                shape=[3, 4], dtype="float32", value=3.0
            )
            paddle.histogram(input=input_value, bins=1, min=5, max=1)

        with self.assertRaises(ValueError):
            self.run_network(net_func)


class TestHistogramOp(OpTest):
    def setUp(self):
        self.op_type = "histogram"
        self.init_test_case()
        np_input = np.random.uniform(low=0.0, high=20.0, size=self.in_shape)
        self.python_api = paddle.histogram
        self.inputs = {"X": np_input}
        self.init_attrs()
        Out, _ = np.histogram(np_input, bins=self.bins, range=(self.min, self.max))
        self.outputs = {"Out": Out.astype(np.int64)}

    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5

    def init_attrs(self):
        self.attrs = {"bins": self.bins, "min": self.min, "max": self.max}

    def test_check_output(self):
        self.check_output()


class TestHistogramOp_ZeroDim(TestHistogramOp):
    def init_test_case(self):
        self.in_shape = []
        self.bins = 5
        self.min = 1
        self.max = 5


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
