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
import paddle.base as base
import paddle.static.amp.amp_nn as amp_nn

paddle.enable_static()
SEED = 2021


def update_loss_scaling_wrapper(
    x,
    found_inf,
    prev_loss_scaling,
    num_good_steps,
    num_bad_steps,
    incr_every_n_steps,
    decr_every_n_nan_or_inf,
    incr_ratio,
    decr_ratio,
    stop_update=False,
):
    amp_nn.update_loss_scaling(
        [x],
        found_inf,
        prev_loss_scaling,
        num_good_steps,
        num_bad_steps,
        incr_every_n_steps,
        decr_every_n_nan_or_inf,
        incr_ratio,
        decr_ratio,
        stop_update,
    )
    return x, prev_loss_scaling, num_good_steps, num_bad_steps


class TestUpdateLossScalingOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "update_loss_scaling"
        self.init()
        found_inf = np.array([False], dtype=np.bool_)
        x = np.random.random((1024, 1024)).astype(self.dtype)

        self.inputs = {
            "X": [("x0", x)],
            "FoundInfinite": found_inf,
            "PrevLossScaling": self.prev_loss_scaling,
            "InGoodSteps": self.num_good_steps,
            "InBadSteps": self.num_bad_steps,
        }

        self.outputs = {
            "Out": [("out0", x)],
            "LossScaling": self.prev_loss_scaling * self.incr_ratio,
            "OutGoodSteps": self.zero_steps,
            "OutBadSteps": self.zero_steps,
        }

    def init(self):
        self.incr_ratio = 2.0
        self.decr_ratio = 0.8
        self.dtype = np.float32
        self.prev_loss_scaling = np.array([2048]).astype(self.dtype)
        self.num_good_steps = np.array([999], dtype=np.int32)
        self.num_bad_steps = np.array([1], dtype=np.int32)
        self.zero_steps = np.array([0], dtype=np.int32)
        self.stop_update = np.array([False], dtype=np.bool_)
        self.attrs = {
            "incr_every_n_steps": 1000,
            "decr_every_n_nan_or_inf": 2,
            "incr_ratio": self.incr_ratio,
            "decr_ratio": self.decr_ratio,
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["Out"])


class TestUpdateLossScalingOpBad(TestUpdateLossScalingOp):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "update_loss_scaling"
        self.init()
        self.python_api = update_loss_scaling_wrapper
        self.python_out_sig = [
            "out0",
            "LossScaling",
            "OutGoodSteps",
            "OutBadSteps",
        ]
        found_inf = np.array([True], dtype=np.bool_)
        x = np.random.random((1024, 1024)).astype(self.dtype)
        i = np.random.randint(0, 1024, 1)
        j = np.random.randint(0, 1024, 1)
        x[i[0]][j[0]] = np.inf

        self.inputs = {
            "X": [("x0", x)],
            "FoundInfinite": found_inf,
            "PrevLossScaling": self.prev_loss_scaling,
            "InGoodSteps": self.num_good_steps,
            "InBadSteps": self.num_bad_steps,
            "StopUpdate": self.stop_update,
        }

        self.outputs = {
            "Out": [("out0", np.zeros_like(x))],
            "LossScaling": self.prev_loss_scaling * self.decr_ratio,
            "OutGoodSteps": self.zero_steps,
            "OutBadSteps": self.zero_steps,
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestUpdateLossScalingLayer(unittest.TestCase):
    def loss_scaling_check(self, use_sdaa=True, scope=base.Scope()):
        a = paddle.static.data(name="a", shape=[1024, 1024], dtype="float32")
        b = paddle.static.data(name="b", shape=[512, 128], dtype="float32")
        x = [a, b]
        found_inf = paddle.static.data(name="found_inf", shape=[1], dtype="bool")
        prev_loss_scaling = paddle.static.data(
            name="prev_loss_scaling", shape=[1], dtype="float32"
        )
        num_good_steps = paddle.static.data(
            name="num_good_steps", shape=[1], dtype="int32"
        )
        num_bad_steps = paddle.static.data(
            name="num_bad_steps", shape=[1], dtype="int32"
        )

        a_v = np.random.random([1024, 1024]).astype("float32")
        b_v = np.random.random([512, 128]).astype("float32")
        found_inf_v = np.array([False]).astype("bool")
        prev_loss_scaling_v = np.array([2048]).astype("float32")
        num_good_steps_v = np.array([999], dtype=np.int32)
        num_bad_steps_v = np.array([1], dtype=np.int32)

        incr_every_n_steps = 1000
        decr_every_n_nan_or_inf = 2
        incr_ratio = 2
        decr_ratio = 0.8

        result = amp_nn.update_loss_scaling(
            x,
            found_inf,
            prev_loss_scaling,
            num_good_steps,
            num_bad_steps,
            incr_every_n_steps,
            decr_every_n_nan_or_inf,
            incr_ratio,
            decr_ratio,
            name="update_loss_scaling",
        )

        place = base.CustomPlace("sdaa", 0)
        exe = base.Executor(place)
        with base.scope_guard(scope):
            exe.run(base.default_startup_program())
            result_v = exe.run(
                feed={
                    "a": a_v,
                    "b": b_v,
                    "found_inf": found_inf_v,
                    "prev_loss_scaling": prev_loss_scaling_v,
                    "num_good_steps": num_good_steps_v,
                    "num_bad_steps": num_bad_steps_v,
                },
                fetch_list=[
                    result,
                    x,
                    found_inf,
                    prev_loss_scaling,
                    num_good_steps,
                    num_bad_steps,
                ],
            )

        assert np.array_equal(result_v[0], a_v)
        assert np.array_equal(result_v[1], b_v)
        assert np.array_equal(result_v[0], result_v[2])
        assert np.array_equal(result_v[1], result_v[3])
        assert np.array_equal(result_v[4], found_inf_v)
        assert np.array_equal(result_v[5], prev_loss_scaling_v * incr_ratio)
        assert np.array_equal(result_v[6], np.zeros_like(num_good_steps_v))
        assert np.array_equal(result_v[7], np.zeros_like(num_bad_steps_v))

    def loss_scaling_check_inf(self, use_sdaa=True, scope=base.Scope()):
        a = paddle.static.data(name="a", shape=[1024, 1024], dtype="float32")
        b = paddle.static.data(name="b", shape=[512, 128], dtype="float32")
        x = [a, b]
        found_inf = paddle.static.data(name="found_inf", shape=[1], dtype="bool")
        prev_loss_scaling = paddle.static.data(
            name="prev_loss_scaling", shape=[1], dtype="float32"
        )
        num_good_steps = paddle.static.data(
            name="num_good_steps", shape=[1], dtype="int32"
        )
        num_bad_steps = paddle.static.data(
            name="num_bad_steps", shape=[1], dtype="int32"
        )

        a_v = np.random.random([1024, 1024]).astype("float32")
        b_v = np.random.random([512, 128]).astype("float32")
        i = np.random.randint(0, 1024, 1)
        j = np.random.randint(0, 1024, 1)
        a_v[i[0]][j[0]] = np.inf
        found_inf_v = np.array([True]).astype("bool")
        prev_loss_scaling_v = np.array([2048]).astype("float32")
        num_good_steps_v = np.array([999], dtype=np.int32)
        num_bad_steps_v = np.array([1], dtype=np.int32)

        incr_every_n_steps = 1000
        decr_every_n_nan_or_inf = 2
        incr_ratio = 2
        decr_ratio = 0.8

        result = amp_nn.update_loss_scaling(
            x,
            found_inf,
            prev_loss_scaling,
            num_good_steps,
            num_bad_steps,
            incr_every_n_steps,
            decr_every_n_nan_or_inf,
            incr_ratio,
            decr_ratio,
            name="update_loss_scaling",
        )

        place = base.CustomPlace("sdaa", 0)
        exe = base.Executor(place)
        with base.scope_guard(scope):
            exe.run(base.default_startup_program())
            result_v = exe.run(
                feed={
                    "a": a_v,
                    "b": b_v,
                    "found_inf": found_inf_v,
                    "prev_loss_scaling": prev_loss_scaling_v,
                    "num_good_steps": num_good_steps_v,
                    "num_bad_steps": num_bad_steps_v,
                },
                fetch_list=[
                    result,
                    x,
                    found_inf,
                    prev_loss_scaling,
                    num_good_steps,
                    num_bad_steps,
                ],
            )
        assert np.array_equal(result_v[0], np.zeros_like(a_v))
        assert np.array_equal(result_v[1], np.zeros_like(b_v))
        assert np.array_equal(result_v[2], np.zeros_like(a_v))
        assert np.array_equal(result_v[3], np.zeros_like(b_v))
        assert np.array_equal(result_v[4], found_inf_v)
        assert np.array_equal(result_v[5], prev_loss_scaling_v * decr_ratio)
        assert np.array_equal(result_v[6], np.zeros_like(num_good_steps_v))
        assert np.array_equal(result_v[7], np.zeros_like(num_bad_steps_v))

    def test_loss_scaling_sdaa(self):
        paddle.set_device("sdaa")
        main = base.Program()
        startup = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, startup):
                self.loss_scaling_check(use_sdaa=True)

    def test_loss_scaling_gpu_inf(self):
        paddle.set_device("sdaa")
        main = base.Program()
        startup = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, startup):
                self.loss_scaling_check_inf(use_sdaa=True)


if __name__ == "__main__":
    unittest.main()
