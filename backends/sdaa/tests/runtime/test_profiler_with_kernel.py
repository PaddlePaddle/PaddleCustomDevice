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
from test_profiler import TestProfiler, filter_launch, filter_kernel, filter_dnn_blas

# NOTE(liaotianju): env vars list here does not affect test,
# but to demonstrate the prerequesites.
# For memset being invoked these flags are needed.
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["FLAGS_init_allocated_mem"] = "True"


def filter_memset(event: dict):
    if not isinstance(event, dict):
        return False
    cat = event.get("cat")
    return isinstance(cat, str) and "memset" in cat.lower()


def filter_memcopy(event: dict):
    if not isinstance(event, dict):
        return False
    cat = event.get("cat")
    return isinstance(cat, str) and "memcpy" in cat.lower()


class TestProfilerWithKernel(TestProfiler):
    def test_profiler(self):
        return super().test_profiler()

    def check(self, trace):
        def error_check_filter(event: dict):
            if not isinstance(event, dict):
                return False
            name = event.get("name")
            return isinstance(name, str) and "sdaaGetLastError" in name

        launch_list = list(filter(filter_launch, trace))
        kernel_list = list(filter(filter_kernel, trace))
        memset_list = list(filter(filter_memset, trace))
        memcpy_list = list(filter(filter_memcopy, trace))
        dnn_blas_list = list(filter(filter_dnn_blas, trace))
        self.assertTrue(len(launch_list) > 0)
        self.assertTrue(len(kernel_list) > 0)
        self.assertTrue(len(memset_list) > 0)
        self.assertTrue(len(memcpy_list) > 0)
        self.assertTrue(len(dnn_blas_list) > 0)
        if os.environ.get("FLAGS_sdaa_error_check", False):
            error_check_list = list(filter(error_check_filter, trace))
            self.assertTrue(len(error_check_list) > 0)
