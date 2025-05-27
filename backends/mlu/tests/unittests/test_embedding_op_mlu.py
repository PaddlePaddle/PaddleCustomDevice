# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np
import unittest
from op_test import OpTest, convert_float_to_uint16

# Currently, MLU do not implement embedding kernel, so disable this unittest.
skip_condition = paddle.get_device().startswith("mlu")

paddle.enable_static()
SEED = 2021


def embedding_wrapper(x, w, padding_idx=None, sparse=False):
    if padding_idx is None or padding_idx == -1:
        return paddle._C_ops.embedding(x, w, -1, sparse)
    else:
        padding_idx = padding_idx if padding_idx >= 0 else (w.shape[0] + padding_idx)
        return paddle._C_ops.embedding(x, w, padding_idx, sparse)


def _lookup(weights, ids, flat_ids, op_version="lookup_table"):
    w_shape = weights.shape
    out_shape = (
        list(ids.shape[:-1]) if op_version == "lookup_table" else list(ids.shape)
    )
    out_shape.append(w_shape[-1])
    out = weights[flat_ids].reshape(out_shape)
    return out


def _get_grad(weights, ids, flat_ids, op_version="lookup_table"):
    w_shape = weights.shape
    w_grad = np.zeros((w_shape), dtype=weights.dtype)
    out_shape = (
        list(ids.shape[:-1]) if op_version == "lookup_table" else list(ids.shape)
    )
    out_grad_shape = (np.prod(out_shape), w_shape[-1])
    out_grad = weights[flat_ids].reshape(out_grad_shape)
    for i, idx in enumerate(flat_ids):
        w_grad[idx, :] += out_grad[i]
    return w_grad


@unittest.skipIf(skip_condition, "MLU do not implement c_embedding kernel")
class TestLookupTableV2(OpTest):
    def setUp(self):
        self.set_mlu()
        self.op_type = "lookup_table_v2"
        self.python_api = embedding_wrapper

        self.place = paddle.CustomPlace("mlu", 0)

        self.init_dtype()
        self.init_dims()
        self.init_padding_idx()
        np.random.seed(SEED)
        w = np.random.random([self.vocab, self.dim]).astype(self.dtype)
        x = np.random.randint(0, self.vocab, size=(self.bsz, self.seqlen)).astype(
            self.ids_dtype
        )
        out = w[x]
        if self.padding_idx != -1:
            out[np.squeeze(x == self.padding_idx)] = np.zeros(self.dim)

        self.inputs = {
            "W": OpTest.np_dtype_to_base_dtype(w),
            "Ids": OpTest.np_dtype_to_base_dtype(x),
        }
        self.attrs = {
            "is_sparse": False,
            "is_distributed": False,
            "remote_prefetch": False,
            "padding_idx": self.padding_idx,
        }
        self.outputs = {"Out": out}

    def set_mlu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32
        self.ids_dtype = np.int32

    def init_dims(self):
        self.bsz = 6
        self.seqlen = 8
        self.vocab = 10
        # embedding_dim is not multiple of 32
        self.dim = 20

    def init_padding_idx(self):
        self.padding_idx = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["W"],
                "Out",
                max_relative_error=0.01,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["W"],
                "Out",
                numeric_place=paddle.CPUPlace(),
            )


@unittest.skipIf(skip_condition, "MLU do not implement c_embedding kernel")
class TestLookupTableV2BF16Op(OpTest):
    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("mlu", 0)

    def init_test(self):
        self.set_mlu()
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = 4
        self.mkldnn_data_type = "bfloat16"

    def setUp(self):
        self.init_test()
        self.dtype = np.uint16

        table = np.random.random((17, 31)).astype("float32")
        self.ids = np.random.randint(0, 17, self.ids_shape).astype("int64")
        self.flat_ids = self.ids.flatten()

        self.w_bf16 = convert_float_to_uint16(table)
        self.out_bf16 = _lookup(self.w_bf16, self.ids, self.flat_ids, self.op_type)
        self.out_fp32 = _lookup(table, self.ids, self.flat_ids, self.op_type)
        self.w_grad_fp32 = _get_grad(table, self.ids, self.flat_ids, self.op_type)

        self.inputs = {"W": self.w_bf16, "Ids": self.ids}
        self.outputs = {"Out": self.out_fp32}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["W"],
            "Out",
            no_grad_set=set("Ids"),
            check_dygraph=False,
            max_relative_error=1.5e-2,
            user_defined_grads=[self.w_grad_fp32],
            user_defined_grad_outputs=[self.out_bf16],
        )


class TestLookupTableV2BF16OpIds3D(TestLookupTableV2BF16Op):
    def init_test(self):
        self.set_mlu()
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = (2, 4)
        self.mkldnn_data_type = "bfloat16"


class TestLookupTableV2FP16(TestLookupTableV2):
    no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16
        self.ids_dtype = np.int32

    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True


class TestLookupTableV2Dim32(TestLookupTableV2):
    def init_dims(self):
        self.bsz = 6
        self.seqlen = 8
        self.vocab = 10
        # embedding_dim is multiple of 32
        self.dim = 64


class TestLookupTableV2Dim32FP16(TestLookupTableV2):
    no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16
        self.ids_dtype = np.int64  # np.int64

    def init_dims(self):
        self.bsz = 6
        self.seqlen = 8
        self.vocab = 10
        self.dim = 64

    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True


class TestLookupTableV2WithPadding(TestLookupTableV2):
    def init_padding_idx(self):
        self.padding_idx = np.random.randint(0, self.vocab)


class TestLookupTableV2WithPadding1(TestLookupTableV2):
    def init_padding_idx(self):
        self.padding_idx = np.random.randint(0, self.vocab)

    def init_dtype(self):
        self.dtype = np.float32
        self.ids_dtype = np.int64  # np.int64


if __name__ == "__main__":
    unittest.main()
