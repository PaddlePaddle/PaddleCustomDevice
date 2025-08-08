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

import copy
import unittest

import numpy as np

import paddle


def compute_index_put_ref(x_np, indices_np, value_np, accumulate=False):
    if accumulate:
        x_np[indices_np] += value_np
        return x_np
    else:
        x_np[indices_np] = value_np
        return x_np


def raw_index_put(x, indices, value, accummulate):
    return paddle.index_put(x, indices, value, accummulate)


def has_duplicate_index(indices, shapes):
    bd_shape = np.broadcast_shapes(*shapes)
    bd_indices = [
        list(np.broadcast_to(indice, bd_shape).flatten()) for indice in indices
    ]

    zip_res = list(zip(*bd_indices))
    if len(zip_res) == len(set(zip_res)):
        return False
    else:
        return True


def gen_indices_np(x_shape, indices_shapes, index_type, is_all_false):
    indices = []
    if index_type == np.bool_:
        indice = np.zeros(indices_shapes[0], dtype=np.bool_)
        if not is_all_false:
            indice.flatten()
            for i in range(len(indice)):
                indice[i] = (i & 1) == 0
            indice = indice.reshape(indices_shapes[0])
        indices.append(indice)
    else:
        while True:
            indices = []
            for i in range(len(indices_shapes)):
                np.random.seed()
                index_np = np.random.randint(
                    low=0,
                    high=x_shape[i],
                    size=indices_shapes[i],
                    dtype=index_type,
                )
                indices.append(index_np)
            if not has_duplicate_index(
                copy.deepcopy(indices), copy.deepcopy(indices_shapes)
            ):
                break
    return tuple(indices)


class TestIndexPutAPIBase(unittest.TestCase):
    def setUp(self):
        self.mixed_indices = False
        self.is_all_false = False
        self.init_dtype_type()
        self.setPlace()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.value_np = np.random.random(self.value_shape).astype(self.dtype_np)

        if self.mixed_indices:
            tmp_indices_np1 = gen_indices_np(
                self.x_shape,
                self.indices_shapes,
                self.index_type_np,
                self.is_all_false,
            )
            tmp_indices_np2 = gen_indices_np(
                self.x_shape,
                self.indices_shapes1,
                self.index_type_np1,
                self.is_all_false,
            )
            self.indices_np = tuple(list(tmp_indices_np1) + list(tmp_indices_np2))
        else:
            self.indices_np = gen_indices_np(
                self.x_shape,
                self.indices_shapes,
                self.index_type_np,
                self.is_all_false,
            )

    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float32"
        self.index_type_pd = "int64"
        self.accumulate = False

    def setPlace(self):
        self.place = ["sdaa"]

    def test_dygraph_forward(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
            self.value_pd = paddle.to_tensor(self.value_np, dtype=self.dtype_pd)
            self.indices_pd = [paddle.to_tensor(indice) for indice in self.indices_np]
            self.indices_pd = tuple(self.indices_pd)
            ref_res = compute_index_put_ref(
                self.x_np, self.indices_np, self.value_np, self.accumulate
            )
            pd_res = paddle.index_put(
                self.x_pd, self.indices_pd, self.value_pd, self.accumulate
            )
            if self.dtype_np != np.float16:
                np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)
            else:
                np.testing.assert_allclose(
                    ref_res, pd_res.numpy(), atol=1e-3, rtol=1e-3
                )


class TestIndexPutAPI1(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float32"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPI2(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (16, 16), (16, 16))
        self.value_shape = (16, 16)
        self.dtype_pd = "float32"
        self.index_type_pd = "int64"
        self.accumulate = False


class TestIndexPutAPI3(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (16, 16), (16, 16))
        self.value_shape = (16, 16)
        self.dtype_pd = "float32"
        self.index_type_pd = "int64"
        self.accumulate = True


class TestIndexPutAPI4(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (16, 16))
        self.value_shape = (56,)
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI5(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 56, 56)
        self.indices_shapes = ((16, 16), (16, 16), (16, 16))
        self.value_shape = (56,)
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI6(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float16"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI7(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "float16"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI8(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int64"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI9(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int64"
        self.index_type_pd = "int32"
        self.accumulate = True


class TestIndexPutAPI10(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int32"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI11(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int32
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "int32"
        self.index_type_pd = "int32"
        self.accumulate = False


class TestIndexPutAPI12(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.uint16
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = [(21,), (21,)]
        self.value_shape = (21,)
        self.dtype_pd = "bfloat16"
        self.index_type_pd = "int64"
        self.accumulate = False


class TestIndexPutAPI13(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110, 94)]
        self.value_shape = (5170,)
        self.dtype_pd = "float32"
        self.index_type_pd = "bool"
        self.accumulate = False


class TestIndexPutAPI14(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110, 94)]
        self.value_shape = (5170,)
        self.dtype_pd = "float16"
        self.index_type_pd = "bool"
        self.accumulate = True


class TestIndexPutAPI15(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.uint16
        self.index_type_np = np.bool_
        self.x_shape = (110, 94)
        self.indices_shapes = [(110,)]
        self.value_shape = (55, 94)
        self.dtype_pd = "bfloat16"
        self.index_type_pd = "bool"
        self.accumulate = True


class TestIndexPutAPIMixedIndices(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 32, 56)
        self.indices_shapes = ((16, 16), (16, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"
        self.accumulate = False

        self.mixed_indices = True
        self.index_type_np1 = np.bool_
        self.indices_shapes1 = [(32,)]
        self.index_type_pd1 = "bool"


class TestIndexPutAPIMixedIndices1(TestIndexPutAPIBase):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int32
        self.x_shape = (110, 42, 32, 56)
        self.indices_shapes = ((16, 16), (16, 16))
        self.value_shape = (16, 16, 56)
        self.dtype_pd = "float32"
        self.index_type_pd = "int32"
        self.accumulate = True

        self.mixed_indices = True
        self.index_type_np1 = np.bool_
        self.indices_shapes1 = [(32,)]
        self.index_type_pd1 = "bool"


if __name__ == "__main__":
    unittest.main()
