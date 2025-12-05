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
/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_exhaustive_search
 * Since Version: 1.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: Represents whether an exhaustive search method is used to
 *       select a convolution algorithm. There are two search methods in cuDNN,
 *       heuristic search and exhaustive search. Exhaustive search attempts
 *       all cuDNN algorithms to select the fastest. This method is very
 *       time-consuming, and the selected algorithm will be cached for a given
 *       layer specification. Once you change the layer specifications
 *       (such as batch size, feature map size), it will search again.
 */

static constexpr int kDefaultConvWorkspaceSizeLimitMB = 512;
/**
 * CUDA related FLAG
 * Name: FLAGS_cublaslt_exhaustive_search_times
 * Since Version: 2.3.0
 * Value Range: int64_t, default=0
 * Example:
 * Note: Represents times of exhaustive search to evaluate performance of
 *       cuBlasLt matmul algorithm (with/without epilogue). Set this flag
 *       with value > 0 to enable exhaustive search. Default is 0, means
 *       getting algorithms via heuristic search. There are two search methods
 *       in cuBlasLt, heuristic search and exhaustive search. Exhaustive search
 *       attempts all cuBlasLt algorithms to select the fastest, which is very
 *       time-consuming, and the selected algorithm will be cached for a given
 *       layer specification Once you change the layer specifications
 *       (such as M, N and K), it will re-search again.
 */
PHI_DEFINE_EXPORTED_int64(
    cublaslt_exhaustive_search_times,
    0,
    "The times of exhaustive search for cuBlasLt matmul with/without "
    " epilogue algorithms, default is 0, means disabling exhaustive search.");

PHI_DEFINE_EXPORTED_bool(
    cudnn_exhaustive_search,
    false,
    "Whether enable exhaustive search for cuDNN convolution or "
    "not, default is False.");

PHI_DEFINE_EXPORTED_bool(
    cudnn_batchnorm_spatial_persistent,
    false,
    "Whether enable CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode for cudnn "
    "batch_norm, default is False.");

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

PHI_DEFINE_EXPORTED_string(
    selected_gpus,
    "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (GPU). If you want to use "
    "all visible devices, set this to empty string. NOTE: the "
    "reason of doing this is that we want to use P2P communication"
    "between GPU devices, use CUDA_VISIBLE_DEVICES can only use"
    "share-memory only.");

PHI_DEFINE_EXPORTED_bool(use_fast_math,
                         false,
                         "Whether to use fast math GPU functions.");

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
/**
 * FlashAttention related FLAG
 * Name: FLAGS_flash_attn_version
 * Value Range: int32, default=2
 * Example:
 * Note: Specify the version of FlashAttention to use, options are 2 or 3.
 *        Version 2 requires Ampere architecture or higher,
 *        while version 3 requires Hopper architecture.
 */
PHI_DEFINE_EXPORTED_int32(
    flash_attn_version,
    2,
    "Specify the version of FlashAttention to use, options are 2 or 3. "
    "Version 2 requires Ampere architecture or higher, "
    "while version 3 requires Hopper architecture.");
#endif
