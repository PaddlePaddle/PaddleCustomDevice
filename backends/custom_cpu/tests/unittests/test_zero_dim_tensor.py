#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest

paddle.set_device("custom_cpu")

unary_api_list = [
    paddle.nn.functional.relu,
]


# Use to test zero-dim in unary API.
class TestUnaryAPI(unittest.TestCase):
    def test(self):
        paddle.disable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        for api in unary_api_list:
            x = paddle.rand([])
            x.stop_gradient = False
            out = api(x)
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            self.assertEqual(x.grad.shape, [])
            self.assertEqual(out.grad.shape, [])

        paddle.enable_static()


reduce_api_list = [
    paddle.sum,
    paddle.mean,
    paddle.min,
    paddle.max,
]


# Use to test zero-dim of reduce API
class TestReduceAPI(unittest.TestCase):
    def test(self):
        paddle.disable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        for api in reduce_api_list:
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, []).astype("bool")
                out = api(x, None)
                self.assertEqual(x.shape, [])
                self.assertEqual(out.shape, [])
            else:
                x = paddle.rand([])
                x.stop_gradient = False
                out = api(x, None)
                out.backward()

                self.assertEqual(x.shape, [])
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.shape, [])
                self.assertEqual(out.grad.shape, [])

        paddle.enable_static()


binary_api_list = [
    {"func": paddle.add, "cls_method": "__add__"},
    {"func": paddle.multiply, "cls_method": "__mul__"},
]

binary_api_list_without_grad = [
    {"func": paddle.equal, "cls_method": "__eq__"},
    {"func": paddle.not_equal, "cls_method": "__ne__"},
    {"func": paddle.greater_equal, "cls_method": "__ge__"},
    {"func": paddle.greater_than, "cls_method": "__gt__"},
    {"func": paddle.less_equal, "cls_method": "__le__"},
    {"func": paddle.less_than, "cls_method": "__lt__"},
]


# Use to test zero-dim of binary API
class TestBinaryAPI(unittest.TestCase):
    def test(self):
        paddle.disable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        for api in binary_api_list + binary_api_list_without_grad:
            # 1) x/y is 0D
            x = paddle.rand([])
            y = paddle.rand([])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api["func"](x, y)
                out_cls = getattr(paddle.Tensor, api["cls_method"])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            self.assertEqual(out.shape, [])
            if api not in binary_api_list_without_grad:
                out.backward()
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [])

            # 2) x is not 0D , y is 0D
            x = paddle.rand([2, 3, 4])
            y = paddle.rand([])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api["func"](x, y)
                out_cls = getattr(paddle.Tensor, api["cls_method"])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            self.assertEqual(out.shape, [2, 3, 4])
            if api not in binary_api_list_without_grad:
                out.backward()
                self.assertEqual(x.grad.shape, [2, 3, 4])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 3) x is 0D , y is not 0D
            x = paddle.rand([])
            y = paddle.rand([2, 3, 4])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api["func"](x, y)
                out_cls = getattr(paddle.Tensor, api["cls_method"])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            self.assertEqual(out.shape, [2, 3, 4])
            if api not in binary_api_list_without_grad:
                out.backward()
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [2, 3, 4])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 4) x is 0D , y is scalar
            x = paddle.rand([])
            y = 0.5
            x.stop_gradient = False
            if isinstance(api, dict):
                out = getattr(paddle.Tensor, api["cls_method"])(x, y)
                self.assertEqual(out.shape, [])

        paddle.enable_static()


# Use to test zero-dim of Sundry API, which is unique and can not be classified
# with others. It can be implemented here flexibly.
class TestSundryAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x = paddle.rand([])

    def test_reshape_list(self):
        x = paddle.rand([])
        x.stop_gradient = False

        out = paddle.reshape(x, [])
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        out = paddle.reshape(x, [1])
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        out = paddle.reshape(x, [-1])
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        out = paddle.reshape(x, [-1, 1])
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1, 1])
        self.assertEqual(out.grad.shape, [1, 1])

    def test_reshape_tensor(self):
        x = paddle.rand([1, 1])
        x.stop_gradient = False

        out = paddle.reshape(x, [])
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        new_shape = paddle.full([1], 1, "int32")
        out = paddle.reshape(x, new_shape)
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        new_shape = paddle.full([1], -1, "int32")
        out = paddle.reshape(x, new_shape)
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        new_shape = [paddle.full([], -1, "int32"), paddle.full([], 1, "int32")]
        out = paddle.reshape(x, new_shape)
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [1, 1])
        self.assertEqual(out.grad.shape, [1, 1])


if __name__ == "__main__":
    unittest.main()
