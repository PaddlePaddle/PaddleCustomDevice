// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/common/flags.h"

// NOTE(zhiqiu): better to share the flags, otherwise we will have too many
// flags.

/**
 * CUDA related related FLAG
 * Name: FLAGS_enable_cublas_tensor_op_math
 * Since Version: 1.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use Tensor Core, faster but it may loss precision.
 */
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

/**
 * CUDA related related FLAG
 * Name: FLAGS_gemm_use_half_precision_compute_type
 * Since Version: 2.4
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use fp16 compute type when the input and output is fp16,
 * faster but it may loss precision.
 */
PHI_DEFINE_EXPORTED_bool(
    gemm_use_half_precision_compute_type,
    false,
    "Whether to use fp16 compute type when the input and output is fp16, "
    "faster but it may loss precision in most case. If true, the compute "
    "type will be set to fp16. Default is false.");

/**
 * CUDA related FLAG
 * Name: FLAGS_selected_gpus
 * Since Version: 1.3.0
 * Value Range: integer list separated by comma, default empty list
 * Example: FLAGS_selected_gpus=0,1,2,3,4,5,6,7 to train or predict with 0~7 gpu
 * cards
 * Note: A list of device ids separated by comma, like: 0,1,2,3
 */
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

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_deterministic
 * Since Version: 0.13.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use deterministic algorithm in cudnn.
 *       If true, it will slow down some operators such as conv and pooling.
 */
PHI_DEFINE_EXPORTED_bool(
    cudnn_deterministic,
    false,
    "Whether allow using an autotuning algorithm for convolution "
    "operator. The autotuning algorithm may be non-deterministic. If "
    "true, the algorithm is deterministic.");

/**
 * CUDA related FLAG
 * Name: FLAGS_embedding_deterministic
 * Since Version: 2.5
 * Value Range: int64, default=0
 * Example:
 * Note: whether to use deterministic algorithm in embedding op.
 *       If it is 1, it will use the optimized deterministic CUDA kernel in
 *       embedding op. If it is 2, it will use the legacy deterministic
 *       CUDA kernel in embedding op.
 */
PHI_DEFINE_EXPORTED_int64(
    embedding_deterministic,
    0,
    "Whether allow using an deterministic algorithm for embedding "
    "operator. The deterministic algorithm may be slower. If "
    "it is larger than 0, the algorithm is deterministic.");

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
PHI_DEFINE_EXPORTED_bool(
    cudnn_exhaustive_search,
    true,
    "Whether enable exhaustive search for cuDNN convolution or "
    "not, default is False.");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_exhaustive_search_times
 * Since Version:
 * Value Range:
 * Example:
 * Note: only used to predict for advanced developer
 */
PHI_DEFINE_EXPORTED_int64(cudnn_exhaustive_search_times,
                          -1,
                          "Exhaustive search times for cuDNN convolution, "
                          "default is -1, not exhaustive search");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_batchnorm_spatial_persistent
 * Since Version: 1.4.0
 * Value Range: bool, default=false
 * Example:
 * Note: CUDNN_BATCHNORM_SPATIAL_PERSISTENT in batchnorm. This mode can be
 * faster in
 *       some tasks because an optimized path may be selected for
 * CUDNN_DATA_FLOAT
 *       and CUDNN_DATA_HALF data types, compute capability 6.0 or higher. The
 *       reason we set it to false by default is that this mode may use scaled
 *       atomic integer reduction that may cause a numerical overflow for
 * certain
 *       input data range.
 */
PHI_DEFINE_EXPORTED_bool(
    cudnn_batchnorm_spatial_persistent,
    true,
    "Whether enable CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode for cudnn "
    "batch_norm, default is False.");

/**
 * NCCL related FLAG
 * Name: FLAGS_sync_nccl_allreduce
 * Since Version: 1.3
 * Value Range: bool, default=true
 * Example:
 * Note: asynchronous nccl allreduce or synchronous issue:
 *       https://github.com/PaddlePaddle/Paddle/issues/15049
 *       If you want to change this default value, why?(gongwb)
 */
PHI_DEFINE_EXPORTED_bool(
    sync_nccl_allreduce,
    true,
    "If set true, will call `cudaStreamSynchronize(nccl_stream)`"
    "after allreduce, this mode can get better performance in some scenarios.");

/**
 * Debug related FLAG
 * Name: check_kernel_launch
 * Since Version: 2.1.0
 * Value Range: bool, default=false
 * Example:
 * Note: Check kernel launch status after every kernel compute.
 */
PHI_DEFINE_EXPORTED_bool(
    check_kernel_launch,
    false,
    "Check kernel launch status after every kernel compute");

/**
 * CUDNN related FLAG
 * Name: conv2d_disable_cudnn
 * Since Version:
 * Value Range: bool, default=false
 * Example:
 * Note: Disable cudnn in conv2d.
 */
PHI_DEFINE_EXPORTED_bool(conv2d_disable_cudnn,
                         false,
                         "Disable cudnn in conv2d");

PHI_DEFINE_EXPORTED_bool(use_fast_math,
                         false,
                         "Whether to use fast math GPU functions.");

PHI_DEFINE_EXPORTED_bool(nccl_blocking_wait, false, "nccl blocking wait");

PHI_DEFINE_EXPORTED_bool(benchmark_nccl,
                         false,
                         "enable nccl debug mode to synchronize nccl comm");

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

PHI_DEFINE_EXPORTED_int32(
    imp_mode,
    1,
    "Whether use the impMode of ixattnbkd for flash attention "
    ", default is CUDNN_FATTN_LEAST_MEM_MODE.");

PHI_DEFINE_EXPORTED_int32(
    causal_mode,
    0,
    "Whether use the causalMode of ixdnn for flash attention "
    ", default is 0.");

PHI_DEFINE_EXPORTED_bool(enable_ixdnn_attn,
                         false,
                         "Whether enable ixdnn for flash attention or "
                         "not, default is False.");

PHI_DEFINE_EXPORTED_bool(enable_ixattnbkd,
                         true,
                         "Whether enable ixattnbkd for flash attention or "
                         "not, default is True.");

PHI_DEFINE_EXPORTED_bool(
    flash_attn_available,
    true,
    "Weather flash attention is available on the current device.");

/**
 * CUDNN related FLAG
 * Name: FLAGS_conv_workspace_size_limit
 * Since Version: 0.13.0
 * Value Range: uint64, default=512 (MB)
 * Example:
 * Note: The internal function of cuDNN obtains the fastest matching algorithm
 *       within this memory limit. Usually, faster algorithms can be chosen in
 *       larger workspaces, but memory space can also be significantly
 * increased.
 *       Users need to balance memory and speed.
 */
PHI_DEFINE_EXPORTED_int64(conv_workspace_size_limit,
                          1024,
                          "cuDNN convolution workspace limit in MB unit.");
