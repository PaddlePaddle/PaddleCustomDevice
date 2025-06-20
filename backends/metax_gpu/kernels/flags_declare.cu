// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// namespace paddle_flags {
// bool FLAGS_cudnn_deterministic = false;
// bool FLAGS_embedding_deterministic = false;
// bool FLAGS_enable_cublas_tensor_op_math = false;
// bool FLAGS_gemm_use_half_precision_compute_type = false;
// bool FLAGS_use_fast_math = false;
// }  // namespace paddle_flags

#include "paddle/common/flags.h"

PHI_DEFINE_EXPORTED_int64(
    embedding_deterministic,
    0,
    "Whether allow using an deterministic algorithm for embedding "
    "operator. The deterministic algorithm may be slower. If "
    "it is larger than 0, the algorithm is deterministic.");
PHI_DEFINE_EXPORTED_bool(
    enable_cublas_tensor_op_math,
    false,
    "The enable_cublas_tensor_op_math indicate whether to use Tensor Core, "
    "but it may loss precision. Currently, There are two CUDA libraries that"
    " use Tensor Cores, cuBLAS and cuDNN. cuBLAS uses Tensor Cores to speed up"
    " GEMM computations(the matrices must be either half precision or single "
    "precision); cuDNN uses Tensor Cores to speed up both convolutions(the "
    "input and output must be half precision) and recurrent neural networks "
    "(RNNs).");
PHI_DEFINE_EXPORTED_bool(
    cudnn_deterministic,
    false,
    "Whether allow using an autotuning algorithm for convolution "
    "operator. The autotuning algorithm may be non-deterministic. If "
    "true, the algorithm is deterministic.");

PHI_DEFINE_EXPORTED_bool(
    gemm_use_half_precision_compute_type,
    false,
    "Whether to use fp16 compute type when the input and output is fp16, "
    "faster but it may loss precision in most case. If true, the compute "
    "type will be set to fp16. Default is false.");

PHI_DEFINE_EXPORTED_bool(use_fast_math,
                         false,
                         "Whether to use fast math GPU functions.");
