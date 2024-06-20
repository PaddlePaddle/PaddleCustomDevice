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
import paddle.base as base
from paddle.base.framework import in_dygraph_mode

SUPPORTED_DTYPES = [np.int32, np.float32, bool]

TEST_META_OP_DATA = [
    {"op_str": "logical_and", "binary_op": True},
    {"op_str": "logical_or", "binary_op": True},
    {"op_str": "logical_xor", "binary_op": True},
    {"op_str": "logical_not", "binary_op": False},
]
TEST_META_SHAPE_DATA = {
    "XDimLargerThanYDim1": {"x_shape": [2, 3, 4, 5], "y_shape": [4, 5]},
    "XDimLargerThanYDim2": {"x_shape": [2, 3, 4, 5], "y_shape": [4, 1]},
    "XDimLargerThanYDim3": {"x_shape": [2, 3, 4, 5], "y_shape": [1, 4, 1]},
    "XDimLargerThanYDim4": {"x_shape": [2, 3, 4, 5], "y_shape": [3, 4, 1]},
    "XDimLargerThanYDim5": {"x_shape": [2, 3, 1, 5], "y_shape": [3, 1, 1]},
    "XDimLessThanYDim1": {"x_shape": [4, 1], "y_shape": [2, 3, 4, 5]},
    "XDimLessThanYDim2": {"x_shape": [1, 4, 1], "y_shape": [2, 3, 4, 5]},
    "XDimLessThanYDim3": {"x_shape": [3, 4, 1], "y_shape": [2, 3, 4, 5]},
    "XDimLessThanYDim4": {"x_shape": [3, 1, 1], "y_shape": [2, 3, 1, 5]},
    "XDimLessThanYDim5": {"x_shape": [4, 5], "y_shape": [2, 3, 4, 5]},
    "Axis1InLargerDim": {"x_shape": [1, 4, 5], "y_shape": [2, 3, 1, 5]},
    "EqualDim1": {"x_shape": [10, 7], "y_shape": [10, 7]},
    "EqualDim2": {"x_shape": [1, 1, 4, 5], "y_shape": [2, 3, 1, 5]},
    "EqualDim3": {"x_shape": [], "y_shape": []},
}

TEST_META_WRONG_SHAPE_DATA = {
    "ErrorDim1": {"x_shape": [2, 3, 4, 5], "y_shape": [3, 4]},
    "ErrorDim2": {"x_shape": [2, 3, 4, 5], "y_shape": [4, 3]},
}


def run_static(x_np, y_np, op_str, use_custom_device=False, binary_op=True):
    paddle.enable_static()
    startup_program = base.Program()
    main_program = base.Program()
    place = paddle.CPUPlace()
    if use_custom_device:
        place = paddle.CustomPlace("sdaa", 0)
    exe = base.Executor(place)
    with base.program_guard(main_program, startup_program):
        x = paddle.static.data(name="x", shape=x_np.shape, dtype=x_np.dtype)
        op = getattr(paddle, op_str)
        feed_list = {"x": x_np}
        if not binary_op:
            res = op(x)
        else:
            y = paddle.static.data(name="y", shape=y_np.shape, dtype=y_np.dtype)
            feed_list["y"] = y_np
            res = op(x, y)
        exe.run(startup_program)
        static_result = exe.run(main_program, feed=feed_list, fetch_list=[res])
    return static_result


def run_dygraph(x_np, y_np, op_str, use_custom_device=False, binary_op=True):
    place = paddle.CPUPlace()
    if use_custom_device:
        place = paddle.CustomPlace("sdaa", 0)
    paddle.disable_static(place)
    op = getattr(paddle, op_str)
    x = paddle.to_tensor(x_np, dtype=x_np.dtype)
    if not binary_op:
        dygraph_result = op(x)
    else:
        y = paddle.to_tensor(y_np, dtype=y_np.dtype)
        dygraph_result = op(x, y)
    return dygraph_result


def np_data_generator(np_shape, dtype, *args, **kwargs):
    if dtype == bool:
        return np.random.choice(a=[True, False], size=np_shape).astype(bool)
    if dtype == np.float16 or dtype == np.float32:
        return np.random.uniform(-100, 100, np_shape).astype(dtype)
    if dtype == np.int32:
        return np.random.randint(-100, 100, np_shape, dtype)


def test(unit_test, use_custom_device=False, test_error=False):
    for op_data in TEST_META_OP_DATA:
        meta_data = dict(op_data)
        meta_data["use_custom_device"] = use_custom_device
        np_op = getattr(np, meta_data["op_str"])
        META_DATA = dict(TEST_META_SHAPE_DATA)
        if test_error:
            META_DATA = dict(TEST_META_WRONG_SHAPE_DATA)
        for shape_data in META_DATA.values():
            for data_type in SUPPORTED_DTYPES:
                if (
                    (
                        (
                            meta_data["op_str"] == "logical_or"
                            or meta_data["op_str"] == "logical_xor"
                        )
                        and data_type != np.int32
                    )
                    or (
                        meta_data["op_str"] == "logical_and" and data_type == np.float32
                    )
                    or (meta_data["op_str"] == "logical_not" and data_type == bool)
                ):
                    pass
                else:
                    meta_data["x_np"] = np_data_generator(
                        shape_data["x_shape"], dtype=data_type
                    )
                    meta_data["y_np"] = np_data_generator(
                        shape_data["y_shape"], dtype=data_type
                    )
                    if meta_data["binary_op"] and test_error:
                        # catch C++ Exception
                        unit_test.assertRaises(BaseException, run_static, **meta_data)
                        unit_test.assertRaises(BaseException, run_dygraph, **meta_data)
                        continue
                    static_result = run_static(**meta_data)
                    dygraph_result = run_dygraph(**meta_data)
                    if meta_data["binary_op"]:
                        np_result = np_op(meta_data["x_np"], meta_data["y_np"])
                    else:
                        np_result = np_op(meta_data["x_np"])
                    unit_test.assertTrue((static_result == np_result).all())
                    unit_test.assertTrue((dygraph_result.numpy() == np_result).all())


def test_type_error(unit_test, use_custom_device, type_str_map):
    def check_type(op_str, x, y, binary_op):
        op = getattr(paddle, op_str)
        error_type = ValueError
        if isinstance(x, np.ndarray):
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            error_type = BaseException
        if binary_op:
            if type_str_map["x"] != type_str_map["y"]:
                unit_test.assertRaises(error_type, op, x=x, y=y)
            if not in_dygraph_mode():
                error_type = TypeError
                unit_test.assertRaises(error_type, op, x=x, y=y, out=1)
        else:
            if not in_dygraph_mode():
                error_type = TypeError
                unit_test.assertRaises(error_type, op, x=x, out=1)

    place = paddle.CPUPlace()
    if use_custom_device:
        place = paddle.CustomPlace("sdaa", 0)
    for op_data in TEST_META_OP_DATA:
        meta_data = dict(op_data)
        binary_op = meta_data["binary_op"]
        if (
            (
                (
                    meta_data["op_str"] == "logical_or"
                    or meta_data["op_str"] == "logical_xor"
                )
                and type_str_map != np.int32
            )
            or (meta_data["op_str"] == "logical_and" and type_str_map == np.float32)
            or (meta_data["op_str"] == "logical_not" and type_str_map == bool)
        ):
            pass
        else:
            paddle.disable_static(place)
            x = np.random.choice(a=[0, 1], size=[10]).astype(type_str_map["x"])
            y = np.random.choice(a=[0, 1], size=[10]).astype(type_str_map["y"])
            check_type(meta_data["op_str"], x, y, binary_op)

            paddle.enable_static()
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[10], dtype=type_str_map["x"])
                y = paddle.static.data(name="y", shape=[10], dtype=type_str_map["y"])
                check_type(meta_data["op_str"], x, y, binary_op)


def type_map_factory():
    return [{"x": np.int32, "y": np.int32}, {"x": np.float32, "y": np.float32}]


class TestSDAA(unittest.TestCase):
    def test(self):
        test(self, True)

    def test_error(self):
        test(self, True, True)

    def test_type_error(self):
        type_map_list = type_map_factory()
        for type_map in type_map_list:
            test_type_error(self, True, type_map)


if __name__ == "__main__":
    unittest.main()
