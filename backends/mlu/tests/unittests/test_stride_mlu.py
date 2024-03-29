# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def ref_view_as_real(x):
    return np.stack([x.real, x.imag], -1)


def ref_view_as_complex(x):
    real, imag = np.take(x, 0, axis=-1), np.take(x, 1, axis=-1)
    return real + 1j * imag


class TestStride(unittest.TestCase):
    def call_transpose(self):
        x_np = np.random.random(size=[2, 3, 4]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # case 1
        # test forward
        x_transposed1 = paddle.transpose(x, perm=[1, 0, 2])
        x_np_transposed1 = x_np.transpose(1, 0, 2)
        np.testing.assert_allclose(x_transposed1.numpy(), x_np_transposed1)

        # test backward
        tmp1 = x_transposed1 * 2
        tmp1.retain_grads()
        loss1 = tmp1.sum()
        loss1.backward(retain_graph=True)
        self.assertTrue((tmp1.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(x_transposed1.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(x_transposed1))
        x_c = x_transposed1.contiguous()
        np.testing.assert_allclose(x_c.numpy(), x_np_transposed1)

        # case 2
        # test forward
        x_transposed2 = paddle.transpose(x_transposed1, perm=[2, 0, 1])
        x_np_transposed2 = x_np_transposed1.transpose(2, 0, 1)
        np.testing.assert_allclose(x_transposed2.numpy(), x_np_transposed2)

        # test backward
        tmp2 = x_transposed2 * 2
        tmp2.retain_grads()
        loss2 = tmp2.sum()
        loss2.backward()
        self.assertTrue((tmp2.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(x_transposed2.is_contiguous())
        y = x_transposed2 + 2
        y_np = x_np_transposed2 + 2
        np.testing.assert_allclose(y.numpy(), y_np)
        self.assertTrue(y.is_contiguous())
        self.assertFalse(x._is_shared_buffer_with(y))

    def call_diagonal(self):
        x_np = np.random.random(size=[2, 3, 4]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.diagonal(x)
        out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
        out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
        out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)

        np_out = np.diagonal(x_np)
        np_out2 = np.diagonal(x_np, offset=0, axis1=2, axis2=1)
        np_out3 = np.diagonal(x_np, offset=1, axis1=0, axis2=1)
        np_out4 = np.diagonal(x_np, offset=0, axis1=1, axis2=2)

        np.testing.assert_allclose(out.numpy(), np_out)
        np.testing.assert_allclose(out2.numpy(), np_out2)
        np.testing.assert_allclose(out3.numpy(), np_out3)
        np.testing.assert_allclose(out4.numpy(), np_out4)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(out.is_contiguous())
        self.assertFalse(out2.is_contiguous())
        self.assertFalse(out3.is_contiguous())
        self.assertFalse(out4.is_contiguous())

        self.assertTrue(x._is_shared_buffer_with(out))
        self.assertTrue(x._is_shared_buffer_with(out2))
        self.assertTrue(x._is_shared_buffer_with(out3))
        self.assertTrue(x._is_shared_buffer_with(out4))

        out_c = out.contiguous()
        out2_c = out2.contiguous()
        out3_c = out3.contiguous()
        out4_c = out4.contiguous()

        np.testing.assert_allclose(out_c.numpy(), np_out)
        np.testing.assert_allclose(out2_c.numpy(), np_out2)
        np.testing.assert_allclose(out3_c.numpy(), np_out3)
        np.testing.assert_allclose(out4_c.numpy(), np_out4)

    def call_slice(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = x[1:10, 0:10, 0:10, 0:20]
        np_out = x_np[1:10, 0:10, 0:10, 0:20]
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_strided_slice(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = x[1:10:2, 0:10:2, 0:10:2, 0:20:2]
        np_out = x_np[1:10:2, 0:10:2, 0:10:2, 0:20:2]
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)

    def call_index_select(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle._C_ops.index_select_strided(x, 5, 3)
        np_out = x_np[:, :, :, 5]
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)

    def call_reshape(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.reshape(x, [10, 100, 20])
        np_out = x_np.reshape(10, 100, 20)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_real(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("complex64")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.real(x)
        np_out = np.real(x_np)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)

    def call_imag(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("complex128")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.imag(x)
        np_out = np.imag(x_np)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)

    def call_as_real(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("complex128")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.as_real(x)
        np_out = ref_view_as_real(x_np)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_as_complex(self):
        x_np = np.random.random(size=[10, 10, 10, 2]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.as_complex(x)
        np_out = ref_view_as_complex(x_np)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_flatten(self):
        x_np = np.random.random(size=[2, 3, 4, 4]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.flatten(x, start_axis=1, stop_axis=2)
        np_out = x_np.reshape(2, 12, 4)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_squeeze(self):
        x_np = np.random.random(size=[5, 1, 10]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.squeeze(x, axis=1)
        np_out = x_np.reshape(5, 10)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_unsqueeze(self):
        x_np = np.random.random(size=[5, 10]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.unsqueeze(x, axis=0)
        np_out = x_np.reshape(1, 5, 10)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_split(self):
        x_np = np.random.random(size=[3, 9, 5]).astype("float32")
        x = paddle.to_tensor(x_np)
        np.testing.assert_allclose(x.numpy(), x_np)

        out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
        np_out0, np_out1, np_out2 = np.split(x_np, 3, 1)

        np.testing.assert_allclose(out0.numpy(), np_out0)
        np.testing.assert_allclose(out1.numpy(), np_out1)
        np.testing.assert_allclose(out2.numpy(), np_out2)

        self.assertFalse(out0.is_contiguous())
        self.assertFalse(out1.is_contiguous())
        self.assertFalse(out2.is_contiguous())

        self.assertTrue(x._is_shared_buffer_with(out0))
        self.assertTrue(x._is_shared_buffer_with(out1))
        self.assertTrue(x._is_shared_buffer_with(out2))

        out0_c = out0.contiguous()
        out1_c = out1.contiguous()
        out2_c = out2.contiguous()

        np.testing.assert_allclose(out0_c.numpy(), np_out0)
        np.testing.assert_allclose(out1_c.numpy(), np_out1)
        np.testing.assert_allclose(out2_c.numpy(), np_out2)

    def call_split2(self):
        x_np = np.random.random(size=[3, 9, 5]).astype("float32")
        x = paddle.to_tensor(x_np)
        np.testing.assert_allclose(x.numpy(), x_np)

        out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
        out = np.split(x_np, [2, 5], 1)
        np_out0 = out[0]
        np_out1 = out[1]
        np_out2 = out[2]

        np.testing.assert_allclose(out0.numpy(), np_out0)
        np.testing.assert_allclose(out1.numpy(), np_out1)
        np.testing.assert_allclose(out2.numpy(), np_out2)

        self.assertFalse(out0.is_contiguous())
        self.assertFalse(out1.is_contiguous())
        self.assertFalse(out2.is_contiguous())

        self.assertTrue(x._is_shared_buffer_with(out0))
        self.assertTrue(x._is_shared_buffer_with(out1))
        self.assertTrue(x._is_shared_buffer_with(out2))

        out0_c = out0.contiguous()
        out1_c = out1.contiguous()
        out2_c = out2.contiguous()

        np.testing.assert_allclose(out0_c.numpy(), np_out0)
        np.testing.assert_allclose(out1_c.numpy(), np_out1)
        np.testing.assert_allclose(out2_c.numpy(), np_out2)

    def call_split3(self):
        x_np = np.random.random(size=[9, 3, 5]).astype("float32")
        x = paddle.to_tensor(x_np)
        np.testing.assert_allclose(x.numpy(), x_np)

        out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=0)
        np_out0, np_out1, np_out2 = np.split(x_np, 3, 0)

        np.testing.assert_allclose(out0.numpy(), np_out0)
        np.testing.assert_allclose(out1.numpy(), np_out1)
        np.testing.assert_allclose(out2.numpy(), np_out2)

        self.assertTrue(out0.is_contiguous())
        self.assertTrue(out1.is_contiguous())
        self.assertTrue(out2.is_contiguous())

        self.assertTrue(x._is_shared_buffer_with(out0))
        self.assertTrue(x._is_shared_buffer_with(out1))
        self.assertTrue(x._is_shared_buffer_with(out2))

        out0_c = out0.contiguous()
        out1_c = out1.contiguous()
        out2_c = out2.contiguous()

        np.testing.assert_allclose(out0_c.numpy(), np_out0)
        np.testing.assert_allclose(out1_c.numpy(), np_out1)
        np.testing.assert_allclose(out2_c.numpy(), np_out2)

        self.assertTrue(out0_c._is_shared_buffer_with(out0))
        self.assertTrue(out1_c._is_shared_buffer_with(out1))
        self.assertTrue(out2_c._is_shared_buffer_with(out2))

    def call_split4(self):
        x_np = np.random.random(size=[9, 3, 5]).astype("float32")
        x = paddle.to_tensor(x_np)
        np.testing.assert_allclose(x.numpy(), x_np)

        out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=0)
        out = np.split(x_np, [2, 5], 0)
        np_out0 = out[0]
        np_out1 = out[1]
        np_out2 = out[2]

        np.testing.assert_allclose(out0.numpy(), np_out0)
        np.testing.assert_allclose(out1.numpy(), np_out1)
        np.testing.assert_allclose(out2.numpy(), np_out2)

        self.assertTrue(out0.is_contiguous())
        self.assertTrue(out1.is_contiguous())
        self.assertTrue(out2.is_contiguous())

        self.assertTrue(x._is_shared_buffer_with(out0))
        self.assertTrue(x._is_shared_buffer_with(out1))
        self.assertTrue(x._is_shared_buffer_with(out2))

        out0_c = out0.contiguous()
        out1_c = out1.contiguous()
        out2_c = out2.contiguous()

        np.testing.assert_allclose(out0_c.numpy(), np_out0)
        np.testing.assert_allclose(out1_c.numpy(), np_out1)
        np.testing.assert_allclose(out2_c.numpy(), np_out2)

        self.assertTrue(out0_c._is_shared_buffer_with(out0))
        self.assertTrue(out1_c._is_shared_buffer_with(out1))
        self.assertTrue(out2_c._is_shared_buffer_with(out2))

    def call_chunk(self):
        x_np = np.random.random(size=[3, 9, 5]).astype("float32")
        x = paddle.to_tensor(x_np)
        np.testing.assert_allclose(x.numpy(), x_np)

        out0, out1, out2 = paddle.chunk(x, chunks=3, axis=1)
        np_out0, np_out1, np_out2 = np.split(x_np, 3, 1)

        np.testing.assert_allclose(out0.numpy(), np_out0)
        np.testing.assert_allclose(out1.numpy(), np_out1)
        np.testing.assert_allclose(out2.numpy(), np_out2)

        self.assertFalse(out0.is_contiguous())
        self.assertFalse(out1.is_contiguous())
        self.assertFalse(out2.is_contiguous())

        self.assertTrue(x._is_shared_buffer_with(out0))
        self.assertTrue(x._is_shared_buffer_with(out1))
        self.assertTrue(x._is_shared_buffer_with(out2))

        out0_c = out0.contiguous()
        out1_c = out1.contiguous()
        out2_c = out2.contiguous()

        np.testing.assert_allclose(out0_c.numpy(), np_out0)
        np.testing.assert_allclose(out1_c.numpy(), np_out1)
        np.testing.assert_allclose(out2_c.numpy(), np_out2)

    def call_unbind(self):
        x_np = np.random.random(size=[3, 9, 5]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out0, out1, out2 = paddle.unbind(x, axis=0)
        np_out0 = x_np[0, 0:100, 0:100]
        np_out1 = x_np[1, 0:100, 0:100]
        np_out2 = x_np[2, 0:100, 0:100]

        np.testing.assert_allclose(out0.numpy(), np_out0)
        np.testing.assert_allclose(out1.numpy(), np_out1)
        np.testing.assert_allclose(out2.numpy(), np_out2)

        # test backward
        tmp = out0 * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out0.is_contiguous())
        self.assertTrue(out1.is_contiguous())
        self.assertTrue(out2.is_contiguous())

        self.assertTrue(x._is_shared_buffer_with(out0))
        self.assertTrue(x._is_shared_buffer_with(out1))
        self.assertTrue(x._is_shared_buffer_with(out2))

        out0_c = out0.contiguous()
        out1_c = out1.contiguous()
        out2_c = out2.contiguous()

        np.testing.assert_allclose(out0_c.numpy(), np_out0)
        np.testing.assert_allclose(out1_c.numpy(), np_out1)
        np.testing.assert_allclose(out2_c.numpy(), np_out2)

        self.assertTrue(out0_c._is_shared_buffer_with(out0))
        self.assertTrue(out1_c._is_shared_buffer_with(out1))
        self.assertTrue(out2_c._is_shared_buffer_with(out2))

    def call_as_strided(self):
        x_np = np.random.random(size=[2, 4, 6]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        np_out = x_np.reshape(8, 6)
        out = paddle.as_strided(x, [8, 6], [6, 1])
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_view(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.view(x, [10, 100, 20])
        np_out = x_np.reshape(10, 100, 20)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_view2(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        out = paddle.view(x, "int32")
        np_out = x_np.view(np.int32)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_view_as(self):
        x_np = np.random.random(size=[10, 10, 10, 20]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        np_out = x_np.reshape(10, 100, 20)
        other = paddle.to_tensor(np_out)
        out = paddle.view_as(x, other)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        tmp = out * 2
        tmp.retain_grads()
        loss = tmp.sum()
        loss.backward()
        self.assertTrue((tmp.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertTrue(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out_c._is_shared_buffer_with(out))

    def call_unfold(self):
        x_np = np.random.random(size=[9]).astype("float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        np.testing.assert_allclose(x.numpy(), x_np)

        # test forward
        np_out = np.stack((x_np[0:2], x_np[4:6]))
        out = paddle.unfold(x, 0, 2, 4)
        np.testing.assert_allclose(out.numpy(), np_out)

        # test backward
        out_b = out * 2
        out_b.retain_grads()
        loss = out_b.sum()
        loss.backward()
        self.assertTrue((out_b.grad.numpy() == 1).all().item())

        # test tensor api
        self.assertFalse(out.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(out))
        out_c = out.contiguous()
        np.testing.assert_allclose(out_c.numpy(), np_out)
        self.assertTrue(out.is_contiguous())
        self.assertFalse(x._is_shared_buffer_with(out))

    def call_stride(self):
        self.call_transpose()
        self.call_diagonal()
        self.call_slice()
        self.call_strided_slice()
        self.call_index_select()
        self.call_reshape()
        self.call_real()
        self.call_imag()
        self.call_as_real()
        self.call_as_complex()
        self.call_flatten()
        self.call_squeeze()
        self.call_unsqueeze()
        # self.call_split()
        # self.call_split2()
        # self.call_split3()
        # self.call_split4()
        # self.call_chunk()
        self.call_unbind()
        self.call_as_strided()
        self.call_view()
        self.call_view2()
        self.call_view_as()
        self.call_unfold()


class TestStrideNPU(TestStride):
    def test_stride_npu(self):
        paddle.set_device("mlu")
        self.call_stride()


class TestToStaticCheck(unittest.TestCase):
    def test_error(self):
        @paddle.jit.to_static(full_graph=True)
        def func():
            x_np = np.random.random(size=[2, 3, 4]).astype("float32")
            x = paddle.to_tensor(x_np)
            y = paddle.transpose(x, perm=[1, 0, 2])
            x.add_(x)

        self.assertRaises(ValueError, func)

    def test_no_error(self):
        @paddle.jit.to_static(full_graph=True)
        def func():
            x_np = np.random.random(size=[2, 3, 4]).astype("float32")
            x = paddle.to_tensor(x_np)
            xx = paddle.assign(x)
            y = paddle.transpose(xx, perm=[1, 0, 2])
            x.add_(x)

        func()


if __name__ == "__main__":
    unittest.main()
