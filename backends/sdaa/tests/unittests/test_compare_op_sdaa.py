#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import unittest

from op_test import OpTest
import paddle


def create_test_class(op_type, typename, callback):
    class Cls(OpTest):
        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = op_type
            self.python_api = eval("paddle." + op_type)

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_output(self):
            paddle.enable_static()
            x = np.random.random(size=(1, 1, 2, 2)).astype(typename)
            y = np.random.random(size=(1, 1, 2, 2)).astype(typename)
            out = callback(x, y)
            self.inputs = {"X": x, "Y": y}
            self.outputs = {"Out": out}
            self.check_output_with_place(place=self.place)

        def test_broadcast_output(self):
            paddle.enable_static()
            x = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(typename)
            y = np.arange(0, 6).reshape((1, 2, 3, 1)).astype(typename)
            out = callback(x, y)
            self.inputs = {"X": x, "Y": y}
            self.outputs = {"Out": out}
            self.check_output_with_place(place=self.place)

    cls_name = "{0}_{1}".format(op_type, typename)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


for _type_name in {"float16", "float32", "int16", "int32", "int64", "bool"}:
    create_test_class("equal", _type_name, lambda _a, _b: _a == _b)
    create_test_class("less_than", _type_name, lambda _a, _b: _a < _b)
    create_test_class("greater_than", _type_name, lambda _a, _b: _a > _b)
    create_test_class("not_equal", _type_name, lambda _a, _b: _a != _b)
    create_test_class("greater_equal", _type_name, lambda _a, _b: _a >= _b)
    create_test_class("less_equal", _type_name, lambda _a, _b: _a <= _b)

if __name__ == "__main__":
    unittest.main()
