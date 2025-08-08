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
import paddle
import paddlenlp_ops

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def ref_set_value_bfai(pre_ids_all, pre_ids, step_idx, stop_flags):
    dim0, dim1 = pre_ids_all.shape

    pre_ids_all.flatten_()
    step_idx.flatten_()
    stop_flags.flatten_()
    pre_ids.flatten_()

    valid_step_idx = paddle.where(step_idx >= 0, step_idx, 0)
    condition = (step_idx >= 0) & (~stop_flags)
    dst_idx = paddle.arange(0, dim0) * dim1 + valid_step_idx
    dst_idx = dst_idx[condition]
    src_idx = paddle.nonzero(condition)
    selected_elements = paddle.gather(pre_ids, src_idx)
    pre_ids_all.scatter_(dst_idx, selected_elements)

    paddle.reshape_(pre_ids_all, [dim0, dim1])
    step_idx.unsqueeze_(axis=-1)
    stop_flags.unsqueeze_(axis=-1)
    pre_ids.unsqueeze_(axis=-1)

    return stop_flags


def clone(*tensors):
    cloned = []
    for tensor in tensors:
        cloned_tensor = tensor.clone()
        cloned.append(cloned_tensor)

    if len(cloned) == 1:
        return cloned[0]
    else:
        return tuple(cloned)


class set_value_afbi_test(unittest.TestCase):
    def is_token_in_targets(self, tokens, target_tensors):
        for target in target_tensors:
            if paddle.all(paddle.equal(tokens, target)):
                return True
        return False

    def test_case1(self):
        pre_ids_all = paddle.full(shape=[4, 96], fill_value=0, dtype="int64")
        pre_ids = paddle.to_tensor([[1000], [2000], [3000], [4000]], dtype="int64")
        step_idx = paddle.to_tensor([[3], [67], [12], [88]], dtype="int64")
        stop_flags = paddle.to_tensor([[False], [False], [False], [False]])
        pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref = clone(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )

        stop_flags = paddlenlp_ops.set_value_by_flags_and_idx(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )
        stop_flags_ref = ref_set_value_bfai(
            pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref
        )

        self.assertTrue(paddle.all(paddle.equal(stop_flags, stop_flags_ref)))
        self.assertTrue(paddle.all(paddle.equal(pre_ids_all, pre_ids_all_ref)))

    def test_case2(self):
        pre_ids_all = paddle.full(shape=[4, 32], fill_value=0, dtype="int64")
        pre_ids = paddle.to_tensor([[1000], [2000], [3000], [4000]], dtype="int64")
        step_idx = paddle.to_tensor([[3], [30], [-2], [28]], dtype="int64")
        stop_flags = paddle.to_tensor([[False], [False], [True], [False]])
        pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref = clone(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )

        stop_flags = paddlenlp_ops.set_value_by_flags_and_idx(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )
        stop_flags_ref = ref_set_value_bfai(
            pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref
        )

        self.assertTrue(paddle.all(paddle.equal(stop_flags, stop_flags_ref)))
        self.assertTrue(paddle.all(paddle.equal(pre_ids_all, pre_ids_all_ref)))

    def test_case3(self):
        pre_ids_all = paddle.full(shape=[16, 256], fill_value=0, dtype="int64")
        pre_ids = paddle.to_tensor(
            [
                [100],
                [200],
                [300],
                [400],
                [500],
                [600],
                [700],
                [800],
                [900],
                [1000],
                [1100],
                [1200],
                [1300],
                [1400],
                [1500],
                [1600],
            ],
            dtype="int64",
        )
        step_idx = paddle.to_tensor(
            [
                [3],
                [30],
                [-2],
                [28],
                [127],
                [255],
                [64],
                [63],
                [200],
                [250],
                [35],
                [188],
                [199],
                [99],
                [100],
                [0],
            ],
            dtype="int64",
        )
        stop_flags = paddle.to_tensor(
            [
                [False],
                [False],
                [True],
                [False],
                [True],
                [False],
                [False],
                [True],
                [True],
                [True],
                [True],
                [False],
                [False],
                [False],
                [True],
                [True],
            ]
        )
        pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref = clone(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )

        stop_flags = paddlenlp_ops.set_value_by_flags_and_idx(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )
        stop_flags_ref = ref_set_value_bfai(
            pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref
        )

        self.assertTrue(paddle.all(paddle.equal(stop_flags, stop_flags_ref)))
        self.assertTrue(paddle.all(paddle.equal(pre_ids_all, pre_ids_all_ref)))

    def test_case4(self):
        pre_ids_all = paddle.to_tensor(
            [
                [1, 2, -1, -1, -1, -1, -1, -1],
                [4, 5, -1, -1, -1, -1, -1, -1],
                [6, 7, -1, -1, -1, -1, -1, -1],
                [8, 9, 10, -1, -1, -1, -1, -1],
            ],
            dtype="int64",
        )
        pre_ids = paddle.to_tensor([[3], [6], [8], [11]], dtype="int64")
        step_idx = paddle.to_tensor([[2], [2], [2], [3]], dtype="int64")
        stop_flags = paddle.to_tensor([[False], [True], [False], [False]])
        pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref = clone(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )

        stop_flags = paddlenlp_ops.set_value_by_flags_and_idx(
            pre_ids_all, pre_ids, step_idx, stop_flags
        )
        stop_flags_ref = ref_set_value_bfai(
            pre_ids_all_ref, pre_ids_ref, step_idx_ref, stop_flags_ref
        )

        self.assertTrue(paddle.all(paddle.equal(stop_flags, stop_flags_ref)))
        self.assertTrue(paddle.all(paddle.equal(pre_ids_all, pre_ids_all_ref)))


if __name__ == "__main__":
    unittest.main()
