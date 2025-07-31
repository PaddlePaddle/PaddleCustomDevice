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
include(version)
include(generic)
include(cblas)
include(flashattn)
include(cutlass)

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

set(THIRD_PARTY_PATH
    "${PADDLE_SOURCE_DIR}/build/third_party"
    CACHE PATH "Third party libraries directory.")

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
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cuda_driver.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_graph.cc
  # Core
  ${PADDLE_SOURCE_DIR}/paddle/phi/core/enforce.cc
  # kernels/Funcs
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/*.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math/*.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/*.cu
  # kernels/gpu
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
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/accuracy_kernel.cu
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
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scale_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/amp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bincount_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cast_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/compare_kerc_idfuncsnel.cu
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
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_unpermute_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/elementwise_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/decode_jpeg_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/nvjpeg.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cupti.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_with_scaled_gradient_grad_kernel_register.cu
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
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_sample_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_sample_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sign_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/split_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/soft_relu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multinomial_kernel.cu
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
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/assign_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/empty_kernel.cc
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
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multinomial_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_kernel.cu
  ${CMAKE_SOURCE_DIR}/kernels/gpudnn/soft.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_kernel.cu
  # kernels/fusion kernels/selected_rows
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adamw_kernel.cu
  # kernels/kps
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/elementwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/elementwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/bitwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/logical_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/reduce_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multiclass_nms3_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/depthwise_conv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/reduce_max_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/array_kernel.cc
  ${CMAKE_SOURCE_DIR}/kernels/funcs/blas/cublas.cc
  ${CMAKE_SOURCE_DIR}/kernels/gpudnn/cudnn.cc
  ${CMAKE_SOURCE_DIR}/kernels/cross_entropy_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/cross_entropy_grad_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/layer_norm_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/layer_norm_grad_kernel_register.cu
  ${CMAKE_SOURCE_DIR}/kernels/flash_attn_grad_kernel.cu
  ${CMAKE_SOURCE_DIR}/kernels/flash_attn_kernel.cu
  ${CMAKE_SOURCE_DIR}/kernels/flashattn.cc)

list(
  REMOVE_ITEM
  CUDA_SRCS
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/gru_compute.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/fc_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/matrix_solve.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/matrix_inverse.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/multihead_matmul_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/softmax.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/weight_only_gemv.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math/context_project.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/fft.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/repeat_tensor2index_tensor.cu)

file(
  GLOB
  ERNIE_CORE_SRCS
  # ernie-core
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/cal_aux_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/cal_aux_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/expand_modality_expert_id_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/int_bincount_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_p_sampling_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_act_kernel.cu)

file(
  GLOB_RECURSE CC_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  runtime/runtime.cc
  kernels/*.cc
  kernels/*.cu
  kernels/cuda_kernels/*.cc
  kernels/cuda_kernels/*.cu
  kernels/ernie_core/*.cu
  kernels/ernie_core/rms_norm_kernel_register.cu
  kernels/ernie_core/top_p_sampling_kernel_register.cu
  kernels/ernie_core/fused_bias_act_kernel_register.cu)

set(CUSTOM_DEVICE_SRCS ${CUDA_SRCS} ${CC_SRCS} ${ERNIE_CORE_SRCS})

set_source_files_properties(${CUSTOM_DEVICE_SRCS} PROPERTIES LANGUAGE CUDA)

set(CMAKE_CUCC_COMPILER "cucc")
set(CMAKE_CUCC_FLAGS "-I /opt/maca/tools/cu-bridge/include/")

set_source_files_properties(
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_kernel.cu
  PROPERTIES LANGUAGE CUDA)
add_library(
  ${TARGET_NAME} SHARED
  ${CUSTOM_DEVICE_SRCS}
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_kernel.cu)

target_include_directories(
  ${TARGET_NAME}
  PRIVATE ${PADDLE_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/kernels
          ${CUDA_INCLUDE_DIRS} ${PADDLE_SOURCE_DIR}/third_party/pybind/include)

target_link_libraries(
  ${TARGET_NAME}
  cutlass
  glog
  eigen3
  gflags
  xxhash
  protobuf
  external_error_proto
  ${PADDLE_CORE_LIB})
target_link_libraries(${TARGET_NAME} /opt/maca/lib/libmccl.so)
target_link_libraries(${TARGET_NAME} /opt/maca/lib/libmcFlashAttn.so)
target_link_libraries(${TARGET_NAME} /opt/maca/lib/libmcpti.so)
include_directories(BEFORE ${PADDLE_SOURCE_DIR})

target_compile_definitions(
  ${TARGET_NAME}
  PUBLIC PADDLE_WITH_CUDA=1
         PADDLE_WITH_CUSTOM_DEVICE=1
         GPUContext=CustomContext
         KPSContext=CustomContext
         STREAM_TYPE=cudaStream_t
         EVENT_TYPE=cudaEvent_t
         EIGEN_USE_GPU=1)

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
