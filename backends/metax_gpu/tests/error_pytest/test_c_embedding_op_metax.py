# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base

paddle.enable_static()


class TestCEmbeddingKernelOpAPI(unittest.TestCase):
    """Test c_embedding_kernel op API."""

    def setUp(self):
        # embedding table: vocab_size=5, embedding_dim=3
        self.vocab_size = 5
        self.embedding_dim = 3
        self.table_np = (
            np.arange(self.vocab_size * self.embedding_dim)
            .reshape([self.vocab_size, self.embedding_dim])
            .astype("float32")
        )
        # index
        self.index_np = np.array([0, 2, 4, 1], dtype="int64")
        self.start_index = 0
        self.vocab_size_param = -1

    def test_static_graph(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            table = paddle.static.data(
                name="table",
                dtype="float32",
                shape=[self.vocab_size, self.embedding_dim],
            )
            index = paddle.static.data(name="index", dtype="int64", shape=[None])
            out = paddle._C_ops.c_embedding(
                table, index, self.start_index, self.vocab_size_param
            )
            place = base.CustomPlace("metax_gpu", 0)
            # if base.core.is_compiled_with_cuda():
            #     place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            res = exe.run(
                main_program,
                feed={"table": self.table_np, "index": self.index_np},
                fetch_list=[out],
            )
            actual = np.array(res[0])
            expected = self.table_np[self.index_np]
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-6,
                err_msg="c_embedding_kernel static graph output is wrong",
            )

    def test_dygraph(self):
        with base.dygraph.guard():
            table = paddle.to_tensor(self.table_np)
            index = paddle.to_tensor(self.index_np)
            out = paddle._C_ops.c_embedding(
                table, index, self.start_index, self.vocab_size_param
            )
            actual = out.numpy()
            expected = self.table_np[self.index_np]
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-6,
                err_msg="c_embedding_kernel dygraph output is wrong",
            )


if __name__ == "__main__":
    unittest.main()
