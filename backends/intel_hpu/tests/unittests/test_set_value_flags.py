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


def ref_set_stop_value_multi_ends(topk_ids, stop_flags, end_ids):
    result_topk_ids = paddle.where(stop_flags, end_ids, topk_ids)
    result_stop_flags = paddle.equal(topk_ids, end_ids) | stop_flags
    return result_topk_ids, result_stop_flags


def is_2d_bool_tensor_equal(a, b):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] and not b[i][j]:
                return False
            if not a[i][j] and b[i][j]:
                return False
    return True


class set_value_flags_test(unittest.TestCase):
    def test_case1(self):
        stop_flags = paddle.to_tensor([[True], [False], [False], [True]])
        end_ids = paddle.to_tensor([[2], [2], [2], [2]], dtype="int64")
        topk_ids = paddle.to_tensor([[3], [3], [2], [2]], dtype="int64")

        topk_ids_out_ref = paddle.to_tensor([[2], [3], [2], [2]], dtype="int64")
        stop_flags_out_ref = paddle.to_tensor([[True], [False], [True], [True]])

        topk_ids_out_ref, stop_flags_out_ref = ref_set_stop_value_multi_ends(
            topk_ids, stop_flags, end_ids
        )

        topk_ids_out, stop_flags_out = paddlenlp_ops.set_stop_value_multi_ends(
            topk_ids, stop_flags, end_ids, 0
        )

        self.assertTrue(paddle.all(paddle.equal(topk_ids_out_ref, topk_ids_out)))
        self.assertTrue(is_2d_bool_tensor_equal(stop_flags_out, stop_flags_out_ref))


if __name__ == "__main__":
    unittest.main()
