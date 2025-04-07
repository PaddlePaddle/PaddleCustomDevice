# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

import unittest

import paddle

import random
import numpy as np

np.random.seed(2024)
paddle.base.set_flags({"FLAGS_use_stride_kernel": True})

all_dtypes = ["int32", "int64", "float16", "float32", "float64"]
cpu_dtypes = ["int32", "int64", "float32", "float64"]


class TestStride(unittest.TestCase):
    def call_transpose(self, dtype):
        x_np = np.random.random(size=[2, 3, 4]).astype(dtype)
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        x_transposed1 = paddle.transpose(x, perm=[1, 0, 2])
        x_np_transposed1 = x_np.transpose(1, 0, 2)
        self.assertTrue(np.allclose(x_transposed1.numpy(), x_np_transposed1))
        self.assertFalse(x_transposed1.is_contiguous())

        x_transposed1 = paddle.abs_(x_transposed1)
        x_np_transposed1 = np.abs(x_np_transposed1)
        self.assertTrue(np.allclose(x_transposed1.numpy(), x_np_transposed1))
        self.assertFalse(x_transposed1.is_contiguous())

        x_c = x_transposed1.contiguous()
        self.assertTrue(np.allclose(x_c.numpy(), x_np_transposed1))

        x_transposed2 = paddle.transpose(x_transposed1, perm=[2, 0, 1])
        x_np_transposed2 = x_np_transposed1.transpose(2, 0, 1)
        self.assertTrue(np.allclose(x_transposed2.numpy(), x_np_transposed2))
        self.assertFalse(x_transposed2.is_contiguous())

        y = x_transposed2 + 2
        y_np = x_np_transposed2 + 2
        self.assertTrue(np.allclose(y.numpy(), y_np))
        self.assertTrue(y.is_contiguous())

    def choose_shape_from_range(self, shape_range: list):
        shape: list = []
        for val in shape_range:
            if isinstance(val, tuple) or isinstance(val, list):
                if len(val) > 1:
                    minval, maxval = val
                    shape.append(random.randint(minval, maxval))
                elif len(val) == 1:
                    shape.append(val[0])
                else:
                    shape.append(0)
            elif val:
                shape.append(val)
            else:
                shape.append(0)

        return shape

    def execute_transpose(self, np_array, perm, device="sdaa"):
        paddle.set_device(device)
        x = paddle.to_tensor(np_array)

        x_transpose = paddle.abs_(paddle.transpose(x, perm=perm))
        x_c = x_transpose.contiguous()

        return x_transpose, x_c

    def call_transpose_01(self, dtype):
        shape_range = [
            [(50, 50), (34, 34)],
            [(1, 64), (1, 58)],
            [(1, 64), (1, 58), (3, 128)],
            [(3, 16), (1, 16), (3, 128), (8, 32)],
            [(3, 16), (1, 16), (3, 128), (8, 32), [8, 16]],
            [(3, 16), (1, 8), (3, 8), (3, 8), [3, 8], [3, 8]],
            [(3, 16), (1, 8), (3, 8), (3, 8), [3, 8], [3, 8], [3, 8]],
            [(3, 7), (1, 8), (3, 8), (3, 8), [3, 8], [3, 8], [3, 8], [2, 3]],
            [(3, 7), (1, 8), (3, 8), (3, 8), [3, 8], [3, 8], [3, 8], [2, 3], [2, 3]],
        ]
        for shape_range in shape_range:
            shape = self.choose_shape_from_range(shape_range)
            x_np = np.random.random(shape).astype(dtype)

            dims = list(range(len(shape)))
            random.shuffle(dims)

            x_t_sdaa, x_c_sdaa = self.execute_transpose(x_np, dims, "sdaa")
            x_t_cpu, x_c_cpu = self.execute_transpose(x_np, dims, "cpu")

            self.assertTrue(np.allclose(x_t_sdaa.numpy(), x_t_cpu.numpy()))
            self.assertEqual(x_t_sdaa.is_contiguous(), x_t_cpu.is_contiguous())

            self.assertTrue(np.allclose(x_c_sdaa.numpy(), x_c_cpu.numpy()))
            self.assertEqual(x_c_sdaa.is_contiguous(), x_c_cpu.is_contiguous())

    def call_transpose_02(self, dtype):
        shape_range = [
            [(3, 16), (1, 16), (3, 128), (8, 32)],
        ]
        all_dims = [
            [0, 3, 1, 2],  # NHWC
            [3, 0, 1, 2],  # CHWN
        ]
        for shape_range in shape_range:
            for dims in all_dims:
                shape = self.choose_shape_from_range(shape_range)
                x_np = np.random.random(shape).astype(dtype)

                dims = list(range(len(shape)))
                random.shuffle(dims)

                x_t_sdaa, x_c_sdaa = self.execute_transpose(x_np, dims, "sdaa")
                x_t_cpu, x_c_cpu = self.execute_transpose(x_np, dims, "cpu")

                self.assertTrue(np.allclose(x_t_sdaa.numpy(), x_t_cpu.numpy()))
                self.assertEqual(x_t_sdaa.is_contiguous(), x_t_cpu.is_contiguous())

                self.assertTrue(np.allclose(x_c_sdaa.numpy(), x_c_cpu.numpy()))
                self.assertEqual(x_c_sdaa.is_contiguous(), x_c_cpu.is_contiguous())

    def call_transpose_03(self, dtype):
        shape_range = [
            [(3, 16), (1, 16), (3, 128)],
        ]
        all_dims = [
            [0, 2, 1],  # NLC
            [2, 0, 1],  # CLN
        ]
        for shape_range in shape_range:
            for dims in all_dims:
                shape = self.choose_shape_from_range(shape_range)
                x_np = np.random.random(shape).astype(dtype)

                dims = list(range(len(shape)))
                random.shuffle(dims)

                x_t_sdaa, x_c_sdaa = self.execute_transpose(x_np, dims, "sdaa")
                x_t_cpu, x_c_cpu = self.execute_transpose(x_np, dims, "cpu")

                self.assertTrue(np.allclose(x_t_sdaa.numpy(), x_t_cpu.numpy()))
                self.assertEqual(x_t_sdaa.is_contiguous(), x_t_cpu.is_contiguous())

                self.assertTrue(np.allclose(x_c_sdaa.numpy(), x_c_cpu.numpy()))
                self.assertEqual(x_c_sdaa.is_contiguous(), x_c_cpu.is_contiguous())

    def call_diagonal(self, dtype):
        x_np = np.random.random(size=[2, 3, 4]).astype(dtype)
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        out = paddle.diagonal(x)
        out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
        out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
        out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)

        out = paddle.abs_(out)
        out2 = paddle.abs_(out2)
        out3 = paddle.abs_(out3)
        out4 = paddle.abs_(out4)

        np_out = np.diagonal(x_np)
        np_out2 = np.diagonal(x_np, offset=0, axis1=2, axis2=1)
        np_out3 = np.diagonal(x_np, offset=1, axis1=0, axis2=1)
        np_out4 = np.diagonal(x_np, offset=0, axis1=1, axis2=2)

        np_out = np.abs(np_out)
        np_out2 = np.abs(np_out2)
        np_out3 = np.abs(np_out3)
        np_out4 = np.abs(np_out4)

        self.assertTrue(np.allclose(out.numpy(), np_out))
        self.assertTrue(np.allclose(out2.numpy(), np_out2))
        self.assertTrue(np.allclose(out3.numpy(), np_out3))
        self.assertTrue(np.allclose(out4.numpy(), np_out4))

        self.assertFalse(out.is_contiguous())
        self.assertFalse(out2.is_contiguous())
        self.assertFalse(out3.is_contiguous())
        self.assertFalse(out4.is_contiguous())

        out_c = out.contiguous()
        out2_c = out2.contiguous()
        out3_c = out3.contiguous()
        out4_c = out4.contiguous()

        self.assertTrue(np.allclose(out_c.numpy(), np_out))
        self.assertTrue(np.allclose(out2_c.numpy(), np_out2))
        self.assertTrue(np.allclose(out3_c.numpy(), np_out3))
        self.assertTrue(np.allclose(out4_c.numpy(), np_out4))

    def call_slice(self, dtype):
        x_np = np.random.random(size=[4, 12, 4]).astype(dtype)
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        out = x[:, :, 0]

        out = paddle.abs_(out)

        np_out = x_np[:, :, 0]

        np_out = np.abs(np_out)

        self.assertTrue(np.allclose(out.numpy(), np_out))

        self.assertFalse(out.is_contiguous())

        out_c = out.contiguous()

        self.assertTrue(np.allclose(out_c.numpy(), np_out))

    def call_strided_slice(self, dtype):
        x_np = np.random.random(size=[4, 12, 10, 20]).astype(dtype)
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        out = x[1:10:2, 0:10:2, 0, 0:20:2]

        out = paddle.abs_(out)

        np_out = x_np[1:10:2, 0:10:2, 0, 0:20:2]

        np_out = np.abs(np_out)

        self.assertTrue(np.allclose(out.numpy(), np_out))

        self.assertFalse(out.is_contiguous())

        out_c = out.contiguous()

        self.assertTrue(np.allclose(out_c.numpy(), np_out))

    def call_index_select(self, dtype):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype(dtype)
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        out = x[:, :, :, 5]
        np_out = x_np[:, :, :, 5]

        out = paddle.abs_(out)
        np_out = np.abs(np_out)

        self.assertTrue(np.allclose(out.numpy(), np_out))

        self.assertFalse(out.is_contiguous())

        out_c = out.contiguous()

        self.assertTrue(np.allclose(out_c.numpy(), np_out))

    def call_D2D_Copy(self, dtype):
        x = np.random.random((11, 121, 1)).astype(dtype)
        x = paddle.to_tensor(x)
        y = paddle.as_strided(x, (11, 121, 1), (121, 1, 121))
        y = y.contiguous()
        self.assertTrue(np.allclose(y.numpy(), x))

    def call_stride(self):
        for dtype in all_dtypes:
            self.call_transpose(dtype)
            self.call_diagonal(dtype)
            self.call_slice(dtype)
            self.call_strided_slice(dtype)
            self.call_index_select(dtype)
            self.call_D2D_Copy(dtype)

        for dtype in cpu_dtypes:
            self.call_transpose_01(dtype)
            self.call_transpose_02(dtype)
            self.call_transpose_03(dtype)


class TestStridesdaa(TestStride):
    def test_stride_sdaa(self):
        paddle.set_device("sdaa")
        self.call_stride()


if __name__ == "__main__":
    unittest.main()
