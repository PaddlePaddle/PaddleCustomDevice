# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License
cmake_minimum_required(VERSION 3.10)
set(PROJ_NAME "paddle-metax-gpu")
project(${PROJ_NAME} CXX C CUDA)

set(TARGET_NAME ${PROJ_NAME})

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PY_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
message(STATUS "Python version detected: ${PY_VERSION}")
set(PYTHON_VERSION ${PY_VERSION})

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
set(WITH_MKLML ON)

include(paddle)
set(THIRD_PARTY_PATH
    "${PADDLE_SOURCE_DIR}/build/third_party"
    CACHE PATH "Third party libraries directory.")

include(version)
include(generic)
include(cblas)
include(flashattn)
include(cutlass)
include(dgc)
include(warpctc)
include(warprnnt)

set(PLUGIN_VERSION ${PADDLE_VERSION})

list(APPEND CMAKE_MODULE_PATH "${PADDLE_SOURCE_DIR}/cmake"
     "${PADDLE_SOURCE_DIR}/cmake/external")

set(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(-std=c++17)

option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_GPU "Compile PaddlePaddle with METAX_GPU" ON)
option(WITH_CUSTOM_DEVICE "Compile PaddlePaddle with CUSTOM_DEVICE" ON)

macro(UNSET_VAR VAR_NAME)
  unset(${VAR_NAME} CACHE)
  unset(${VAR_NAME})
endmacro()

include(cuda)
include(gflags)
include(glog)
include(eigen)
include(xxhash)
include(zlib)
include(protobuf)
include(generate_pb)

set(PROTO_FILE "${PADDLE_SOURCE_DIR}/paddle/phi/core/external_error.proto")
get_filename_component(PROTO_WE "${PROTO_FILE}" NAME_WE)

set(GENERATED_SRC
    "${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core/${PROTO_WE}.pb.cc")
set(GENERATED_HDR
    "${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core/${PROTO_WE}.pb.h")

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "PROTOBUF_PROTOC_EXECUTABLE: ${PROTOBUF_PROTOC_EXECUTABLE}")
message(
  STATUS
    "Full protoc command: ${PROTOBUF_PROTOC_EXECUTABLE} -I${CMAKE_CURRENT_SOURCE_DIR}/paddle/phi/core/ --cpp_out=${CMAKE_CURRENT_BINARY_DIR} ${PROTO_FILE}"
)

add_custom_command(
  OUTPUT "${GENERATED_SRC}" "${GENERATED_HDR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core"
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I${PADDLE_SOURCE_DIR}/paddle/phi/core/
          --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core ${PROTO_FILE}
  DEPENDS "${PROTO_FILE}"
  COMMENT "Generating C++ protocol buffer for ${PROTO_FILE}"
  VERBATIM)

add_library(external_error_proto STATIC "${GENERATED_SRC}")
target_include_directories(external_error_proto
                           PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(external_error_proto PUBLIC protobuf)
set_target_properties(external_error_proto PROPERTIES POSITION_INDEPENDENT_CODE
                                                      ON)

file(
  GLOB
  CUDA_SRCS
  # backends
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_info.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/dynamic_loader.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cublas.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cublasLt.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cudnn.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cuda_driver.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_graph.cc
  # Core
  ${PADDLE_SOURCE_DIR}/paddle/phi/core/enforce.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/core/mixed_vector.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cusparse.cc
  # kernels/Funcs
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/*.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math/*.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/*.cu
  # kernels/gpu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/spectral_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamw_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adagrad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/add_n_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/arange_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adadelta_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/accuracy_check_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/accuracy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/allclose_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/all_gather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/all_reduce_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/all_to_all_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/arg_min_max_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/apply_per_channel_scale_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/as_complex_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/as_real_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/asgd_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/assign_pos_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/amp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ap_facade_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/barrier_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bincount_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/box_clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cast_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/compare_kerc_idfuncsnel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/correlation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dist_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/compare_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/compare_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/numel_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diag_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diag_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/einsum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/einsum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/decode_jpeg_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/nvjpeg.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cupti.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_as_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_as_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eye_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_tensor_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_tensor_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/full_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gaussian_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_put_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_put_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/put_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/one_hot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/randint_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/set_value_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/set_value_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_slice_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_slice_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_copy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slice_grad_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slice_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/take_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/take_along_axis_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diagonal_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diagonal_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logsumexp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_inplace_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_inplace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_random_batch_size_like_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/unsqueeze_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/unsqueeze_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/squeeze_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/squeeze_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sign_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/split_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/soft_relu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mean_all_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mean_all_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multiplex_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multiplex_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pow2_decay_with_linear_warmup_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sigmoid_cross_entropy_with_logits_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_k_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_k_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/where_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/where_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/empty_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lerp_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lerp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/flatten_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/flatten_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_all_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_any_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_sum_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_mean_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reshape_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reshape_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/contiguous_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gelu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gelu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/triu_indices_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_indices_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_triu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_triu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unbind_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/gather_scatter_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/fc_functor.cu
  ${CMAKE_SOURCE_DIR}/kernels/gpudnn/soft.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cusolver.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/collect_fpn_proposals_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/check_numerics_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_split_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/broadcast_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/decayed_adagrad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/debug_tools_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cumprod_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cumprod_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/crop_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/crop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fetch_barrier_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fake_dequantize_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fake_quantize_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erfinv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erfinv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erf_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erf_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/edit_distance_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dgc_clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diag_embed_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dequantize_log_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dequantize_abs_max_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/depend_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fused_adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ftrl_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frobenius_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frobenius_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frame_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frame_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fold_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fold_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fft_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fft_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/huber_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/huber_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/histogram_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/hinge_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/hinge_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gru_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/generate_proposals_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gaussian_inplace_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaln_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/beam_search_decode_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/beam_search_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0e_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0e_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1e_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1e_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/im2sequence_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/im2sequence_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/increment_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_grad_add_to_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy_bwd_w_downcast.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_get_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_get_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_put_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_put_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_sample_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_sample_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_select_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_select_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/inverse_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/isclose_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/isfinite_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/atan2_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/atan2_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/auc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/average_accumulates_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy2_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy2_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_recv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/matrix_rank_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/poisson_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logcumsumexp_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/quant_linear_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/polygamma_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lrn_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/merged_momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/put_along_axis_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/polygamma_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nanmedian_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mode_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pixel_unshuffle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lod_reset_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mode_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/overlap_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/maxout_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_unpack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pixel_shuffle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lod_reset_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/qr_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prelu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prelu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/psroi_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/overlap_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/maxout_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_send_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/psroi_pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pixel_shuffle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_unpack_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nll_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nanmedian_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/margin_cross_entropy_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_fill_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/number_count_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lrn_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/poisson_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nms_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_fill_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prod_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nadam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prune_gate_by_capacity_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prior_box_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logspace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_select_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multinomial_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nll_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_unpermute_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logsumexp_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moving_average_abs_max_scale_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_select_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/solve_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/radam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/random_routing_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/renorm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scale_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/randperm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_as_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_as_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_scatter_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/renorm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roi_align_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roi_align_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roi_pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roi_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roll_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roll_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rprop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rrelu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rrelu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/searchsorted_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/segment_pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/segment_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/selu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_u_recv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_u_recv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_ue_recv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_ue_recv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_uv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_uv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_expand_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_mask_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sgd_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/share_data_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/shard_index_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/shuffle_batch_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/shuffle_batch_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sigmoid_cross_entropy_with_logits_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slogdeterminant_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sparse_momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/straight_through_estimator_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/svd_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sync_batch_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sync_comm_stream_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sync_calc_stream_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/temporal_shift_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/temporal_shift_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trace_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trunc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trunc_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/truncated_gaussian_random_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unfold_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unfold_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unpool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lstsq_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unpool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unstack_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lgamma_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/linspace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kron_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kron_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stack_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unstack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/viterbi_decode_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/warprnnt_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/yolo_box_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kthvalue_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kthvalue_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dgc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaincc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaincc_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/llm_int8_linear_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/baddbmm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/baddbmm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/load_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/load_combine_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_softmax_with_multi_label_cross_entropy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/save_combine_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/save_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dropout_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dropout_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bce_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bce_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/meshgrid_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/meshgrid_grad_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad3d_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad3d_kernel.cu
  # ############################################################################
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/array_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/set_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/is_empty_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/dist_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/coalesce_tensor_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_amin_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/batch_norm_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_variance_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/shape_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_amax_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/prod_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/assign_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reverse_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/full_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strided_slice_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fake_quantize_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_min_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/check_memory_continue_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strided_slice_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/npu_identity_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/activation_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/scale_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/elementwise_multiply_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/full_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/uniform_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/merge_selected_rows_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/isfinite_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/lamb_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/add_n_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/lookup_table_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/ftrl_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/dgc_clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/share_data_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/lookup_table_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/uniform_random_batch_size_like_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/get_tensor_from_selected_rows_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/batch_norm_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/batch_norm_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/empty_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/sparse_utils_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sync_batch_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sparse_utils_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/reshape_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/full_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/transpose_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/elementwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sparse_attention_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/reshape_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/slice_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/slice_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/unary_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/coalesce_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/transpose_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/mask_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sync_batch_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/unary_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/transfer_layout_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/elementwise_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/mask_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/ext_build_src_rank_and_local_expert_id_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_no_weight_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/layer_norm_cuda_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/one_hot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_expand_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_crop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_and_quant_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_ops_partial_nosoftmaxtopk_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_crop_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_permute_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/randint_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_generate_proposals_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/anchor_generator_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_no_weight_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_expand_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_permute_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_ops_partial_nosoftmaxtopk_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/uniform_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_rope_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_dropout_residual_layer_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_seqpool_cvm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_seqpool_cvm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_transpose_split_quant_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_transpose_wlch_split_quant_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_dropout_residual_layer_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/distributed_fused_lamb_init_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_act_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/qkv_unpack_mha_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_softmax_mask_upper_triangle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_softmax_mask_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/masked_multihead_attention_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_softmax_mask_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fusion_group_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/skip_layernorm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/affine_channel_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ap_trivial_fusion_begin_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_layernorm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_embedding_eltwise_layernorm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_stack_transpose_quant_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_rope_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_swiglu_weighted_bwd_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/core/flags.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math_function.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/repeat_tensor2index_tensor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/binomial_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bernoulli_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cufft.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/box_coder_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/broadcast_tensors_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/broadcast_tensors_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/channel_shuffle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/channel_shuffle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/complex_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/complex_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_maxmin_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_maxmin_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/digamma_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/digamma_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dot_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eigh_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eigvalsh_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/exponential_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaincc_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_tree_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/graph_reindex_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/graph_sample_neighbors_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/group_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gumbel_softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gumbel_softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_p_sampling_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lars_momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/partial_sum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/gpu_info.cc
  # ############################################################################
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adamw_kernel.cu
  # kernels/kps
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/elementwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/elementwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/bitwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/logical_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/reduce_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/reduce_max_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/array_kernel.cc
  ${CMAKE_SOURCE_DIR}/kernels/gpudnn/cudnn.cc
  ${CMAKE_SOURCE_DIR}/kernels/metax_context.cc
  ${CMAKE_SOURCE_DIR}/kernels/cross_entropy_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/cross_entropy_grad_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/layer_norm_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/layer_norm_grad_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/flash_attn_grad_kernel.cu
  ${CMAKE_SOURCE_DIR}/kernels/flash_attn_kernel.cu
  ${CMAKE_SOURCE_DIR}/kernels/flashattn.cc)

list(REMOVE_ITEM CUDA_SRCS
     ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/softmax.cu
     ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/weight_only_gemv.cu)

file(
  GLOB
  ERNIE_CORE_SRCS
  # ernie-core
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/cal_aux_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/cal_aux_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/expand_modality_expert_id_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/int_bincount_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_act_kernel.cu)

file(
  GLOB_RECURSE CC_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  runtime/runtime.cc
  passes/*.cc
  common/*.cc
  kernels/*.cc
  kernels/*.cu
  kernels/fusion/*.cc
  kernels/fusion/*.cu
  kernels/gpudnn/*.cc
  kernels/gpudnn/*.cu
  kernels/cuda_kernels/*.cc
  kernels/cuda_kernels/*.cu
  kernels/ernie_core/*.cu)

set(CUSTOM_DEVICE_SRCS ${CUDA_SRCS} ${CC_SRCS} ${ERNIE_CORE_SRCS})

set_source_files_properties(${CUSTOM_DEVICE_SRCS} PROPERTIES LANGUAGE CUDA)
set(MACA_PATH $ENV{MACA_PATH})
set(CMAKE_CUCC_COMPILER "cucc")
set(CMAKE_CUCC_FLAGS "-I ${MACA_PATH}/tools/cu-bridge/include/")

add_library(${TARGET_NAME} SHARED ${CUSTOM_DEVICE_SRCS})

target_include_directories(
  ${TARGET_NAME}
  PRIVATE ${PADDLE_SOURCE_DIR}
          ${CMAKE_SOURCE_DIR}
          ${CMAKE_SOURCE_DIR}/kernels
          ${CUDA_INCLUDE_DIRS}
          ${WARPCTC_INCLUDE_DIR}
          ${WARPRNNT_INCLUDE_DIR}
          ${PADDLE_SOURCE_DIR}/third_party/pybind/include
          ${PADDLE_SOURCE_DIR}/paddle/phi/api/include/compat)

target_link_libraries(
  ${TARGET_NAME}
  cutlass
  glog
  eigen3
  gflags
  xxhash
  protobuf
  external_error_proto
  dgc
  ${WARPCTC_LIBRARIES}
  ${WARPRNNT_LIBRARIES}
  ${PADDLE_CORE_LIB})

target_link_libraries(${TARGET_NAME} ${MACA_PATH}/lib/libmccl.so)
target_link_libraries(${TARGET_NAME} ${MACA_PATH}/lib/libmcFlashAttn.so)
target_link_libraries(${TARGET_NAME} ${MACA_PATH}/lib/libmcpti.so)

include_directories(BEFORE ${PADDLE_SOURCE_DIR})

target_compile_definitions(
  ${TARGET_NAME}
  PUBLIC PADDLE_WITH_CUDA=1
         PADDLE_WITH_CUSTOM_DEVICE=1
         mcblasContext=cublasContext
         cublasLtContext=mcblasLtContext
         GPUContext=CustomContext
         KPSContext=CustomContext
         STREAM_TYPE=cudaStream_t
         EVENT_TYPE=cudaEvent_t
         EIGEN_USE_GPU=1
         CUDA_LIB_NAME="libmcruntime.so"
         BLAS_LIB_NAME="libmcblas.so"
         BLASLT_LIB_NAME="libmcblasLt.so"
         DNN_LIB_NAME="libmcdnn.so"
         PTI_LIB_NAME="libmcpti.so"
         RAND_LIB_NAME="libcurand.so"
         JPEG_LIB_NAME="libnvjpeg.so"
         SOLVER_LIB_NAME="libmcsolver.so"
         SPARSE_LIB_NAME="libmcsparse.so"
         RTC_LIB_NAME="libmcruntime.so"
         FLASHATTN_LIB_NAME="libmcFlashAttn.so"
         FLASHATTNV3_LIB_NAME="libflashattnv3.so"
         CCL_LIB_NAME="libmccl.so"
         FFT_LIB_NAME="libcufft.so"
         SPARSELT_LIB_NAME="libcusparseLt.so"
         CUPTI_LIB_PATH="/root/cu-bridge/CUDA_DIR/extras/CUPTI/lib64")

# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
  TARGET ${TARGET_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating plugin directories------>>>")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${TARGET_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
