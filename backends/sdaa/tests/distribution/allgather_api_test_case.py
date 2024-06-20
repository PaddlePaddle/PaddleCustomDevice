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

import os

import numpy as np
import communication_api_test_base as test_base

import paddle
import paddle.distributed as dist


class StreamAllgatherTestCase:
    def __init__(self):
        self._sync_op = eval(os.getenv("sync_op"))
        self._use_calc_stream = eval(os.getenv("use_calc_stream"))
        self._backend = os.getenv("backend")
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        os.environ["PADDLE_DISTRI_BACKEND"] = self._backend

    def run_test_case(self):
        dist.init_parallel_env()

        test_data_list = []
        for seed in self._seeds:
            test_data_list.append(
                test_base.create_test_data(
                    shape=self._shape, dtype=self._dtype, seed=seed
                )
            )

        rank = dist.get_rank()
        tensor = paddle.to_tensor(test_data_list[rank])

        # case 1: pass an empty tensor list
        empty_tensor_list = []
        task = dist.stream.all_gather(
            empty_tensor_list,
            tensor,
            sync_op=self._sync_op,
            use_calc_stream=self._use_calc_stream,
        )
        if not self._sync_op:
            task.wait()
        np.testing.assert_allclose(
            empty_tensor_list, test_data_list, rtol=1e-05, atol=1e-05
        )

        # case 2: pass a pre-sized tensor list
        full_tensor_list = [paddle.empty_like(tensor) for _ in test_data_list]
        task = dist.stream.all_gather(
            full_tensor_list,
            tensor,
            sync_op=self._sync_op,
            use_calc_stream=self._use_calc_stream,
        )
        if not self._sync_op:
            task.wait()
        np.testing.assert_allclose(
            full_tensor_list, test_data_list, rtol=1e-05, atol=1e-05
        )

        # case 3: pass a pre-sized tensor
        result_tensor = paddle.concat(
            [paddle.to_tensor(data) for data in test_data_list]
        )
        out_tensor = paddle.empty_like(result_tensor)
        task = dist.stream.all_gather(
            out_tensor,
            tensor,
            sync_op=self._sync_op,
            use_calc_stream=self._use_calc_stream,
        )
        if not self._sync_op:
            task.wait()
        np.testing.assert_allclose(out_tensor, result_tensor, rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    StreamAllgatherTestCase().run_test_case()
