# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from test_parallel_dygraph_mp_layers import TestMultipleCustomDevices


class TestHybridPipeParallel(TestMultipleCustomDevices):
    # def test_hybrid_parallel_pp_layer(self):
    #     self.run_mnist_2_custom_devices(
    #         os.path.abspath('hybrid_parallel_pp_layer.py'), 'mlu'
    #     )

    # def test_hybrid_parallel_pp_tuple_imluts(self):
    #     self.run_mnist_2_custom_devices(
    #         os.path.abspath('hybrid_parallel_pp_embedding.py'), 'mlu'
    #     )

    def test_hybrid_parallel_shared_weight(self):
        self.run_mnist_2_custom_devices(
            "model_parallel/hybrid_parallel_shared_weight.py", "mlu"
        )

    # def test_pipeline_parallel_amp(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_amp.py', 'mlu')

    # def test_pipeline_parallel_fp16(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_fp16.py', 'mlu')

    # def test_pipeline_parallel_bf16(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_bf16.py', 'mlu')

    # def test_hybrid_parallel_transformer(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_transformer.py', 'mlu')

    # def test_hybrid_parallel_save_load(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_save_load.py', 'mlu')

    # def test_hybrid_parallel_recompute(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_recompute.py', 'mlu')

    # def test_hybrid_parallel_pp_clip_grad(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_clip_grad.py', 'mlu')

    # def test_hybrid_parallel_transformer_unbalanced_data(self):
    #     self.run_mnist_2_custom_devices('hybrid_parallel_pp_transformer_unbalanced_data.py', 'mlu')


if __name__ == "__main__":
    unittest.main()
