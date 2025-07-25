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
set(PROJ_NAME "paddle-iluvatar-gpu")
project(${PROJ_NAME} CXX C CUDA)

set(TARGET_NAME ${PROJ_NAME})

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
include(external/eigen)
include(external/xxhash)
include(external/zlib)
include(external/protobuf)

set(PLUGIN_VERSION ${PADDLE_VERSION})
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

add_custom_target(external_deps DEPENDS eigen3 zlib protobuf)

if(WITH_COREX)
  add_definitions(-DPADDLE_WITH_COREX)
  add_definitions(-DEIGEN_USE_COREX)
  add_definitions(-DEIGEN_USE_GPU)
endif()

list(APPEND CMAKE_MODULE_PATH "${PADDLE_SOURCE_DIR}/cmake"
     "${PADDLE_SOURCE_DIR}/cmake/external")

enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/runtime)

add_definitions(-std=c++17)

option(WITH_TESTING "compile with unit testing" OFF)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_GPU "Compile PaddlePaddle with ILUVATAR_GPU" ON)

include(cuda)
include(gflags)
include(glog)

file(
  GLOB
  CUDA_SRCS
  # backends
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_info.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_graph.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cuda_driver.cc
  # Core
  ${PADDLE_SOURCE_DIR}/paddle/phi/core/enforce.cc
  # kernels/funcs
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/concat_and_split_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math_function.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/*.cu
  # cudnn/cublas
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cudnn.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cublas.cc
  # kernels/gpu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamw_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/amp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/arange_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_identity_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cast_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/contiguous_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/einsum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/elementwise_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/full_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gaussian_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gelu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_put_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logsumexp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mean_all_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mean_all_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multinomial_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nonzero_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/numel_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/one_hot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/put_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/randint_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scale_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sign_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slice_grad_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slice_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/split_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_copy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_elementwise_copy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_slice_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/set_value_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/set_value_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/take_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_triu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unbind_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/where_kernel.cu
  # kernels/selected_rows
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adamw_kernel.cu
  # kernels/kps
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/bitwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/elementwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/compare_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/logical_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/reduce_kernel.cu
  # kernels/legacy/kps
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/elementwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/compare_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/reduce_max_kernel.cu
  # kernels
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/assign_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/empty_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/flatten_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/flatten_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_all_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_any_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_max_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_mean_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_sum_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reshape_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reshape_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/shape_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strided_slice_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strided_slice_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/squeeze_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/squeeze_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/unsqueeze_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/unsqueeze_kernel.cc
  # ernie_core
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_act_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_modality_expert_id_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_combine_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_combine_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_gate_dispatch_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_gate_dispatch_permute_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_ops_partial_nosoftmaxtopk_kernel.cu
)

file(
  GLOB_RECURSE CC_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  runtime/runtime.cc
  runtime/iluvatar_context.cc
  common/*.cc
  kernels/cuda_kernels/*.cc
  kernels/cuda_kernels/*.cu
  kernels/ernie_core/*.cu
  kernels/ernie_core/*.cc)

set(CUSTOM_DEVICE_SRCS ${CUDA_SRCS} ${CC_SRCS})

set_source_files_properties(${CUSTOM_DEVICE_SRCS} PROPERTIES LANGUAGE CUDA)

add_library(${TARGET_NAME} SHARED ${CUSTOM_DEVICE_SRCS})
add_dependencies(${TARGET_NAME} external_deps)

target_include_directories(
  ${TARGET_NAME} PRIVATE ${PADDLE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
                         ${CMAKE_SOURCE_DIR}/kernels ${CUDA_INCLUDE_DIRS})

target_link_libraries(
  ${TARGET_NAME}
  PRIVATE glog
          ${PADDLE_CORE_LIB}
          cudart
          eigen3
          gflags
          xxhash
          protobuf
          external_error_proto
          cuinfer
          nccl)

include_directories(BEFORE ${PADDLE_SOURCE_DIR})

target_compile_definitions(
  ${TARGET_NAME}
  PUBLIC PADDLE_WITH_CUDA=1
         PADDLE_WITH_CUSTOM_DEVICE=1
         GPUContext=CustomContext
         STREAM_TYPE=cudaStream_t
         EVENT_TYPE=cudaEvent_t
         KPSContext=CustomContext
         __NVCC__=__IXCC__)

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
