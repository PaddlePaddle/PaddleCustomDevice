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

set(PLUGIN_VERSION "0.0.1")
set(TARGET_NAME ${PROJ_NAME})

set(CMAKE_MODULE_PATH 
    "${CMAKE_SOURCE_DIR}/cmake"
)
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

set(WITH_MKLML ON)

include(paddle)
include(version)
include(generic)
include(cblas)
include(external/eigen)
include(external/zlib)
include(external/protobuf)

add_custom_target(external_deps
  DEPENDS eigen3 zlib protobuf
)

if(WITH_COREX)
add_definitions(-DPADDLE_WITH_COREX)
add_definitions(-DEIGEN_USE_COREX)
add_definitions(-DEIGEN_USE_GPU)
endif()

list(APPEND CMAKE_MODULE_PATH
    "${PADDLE_SOURCE_DIR}/cmake"
    "${PADDLE_SOURCE_DIR}/cmake/external"
)

enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(-std=c++17)

option(WITH_TESTING "compile with unit testing" OFF)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_GPU "Compile PaddlePaddle with ILUVATAR_GPU" ON)

set(THIRD_PARTY_PATH "${PADDLE_SOURCE_DIR}/build/third_party" 
    CACHE PATH "Third party libraries directory.")

include(cuda)
include(gflags)
include(glog)


file(
  GLOB CUDA_SRCS
  # **Dependent files**
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_info.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cuda_driver.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math_function.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cudnn.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/dynload/cublas.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/core/enforce.cc
  # **op list**
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reshape_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reshape_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/shape_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/squeeze_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strided_slice_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_sum_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/unsqueeze_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamw_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/amp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/arange_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/elementwise_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_identity_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cast_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/contiguous_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/einsum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/full_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gaussian_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logsumexp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multinomial_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nonzero_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/numel_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/one_hot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/put_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rms_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scale_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slice_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slice_grad_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/softmax_impl.h
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/split_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/take_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_triu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unbind_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/where_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_put_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/broadcast.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/concat_and_split_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_graph.cc
)

file(
  GLOB_RECURSE CC_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  runtime/runtime.cc
  # **Dependent files**
  common/cuda_flags.cc
  # **op list**
  kernels/reduce_sum_kernel_register.cc
  kernels/empty_kernel.cc
  kernels/flatten_kernel.cc
  kernels/reshape_grad_kernel_register.cc
  kernels/reshape_kernel_register.cc
  kernels/shape_kernel_register.cc
  kernels/unsqueeze_kernel_register.cc
  kernels/squeeze_kernel_register.cc
  kernels/strided_slice_kernel_register.cc
  kernels/kps/bitwise_kernel.cu
  kernels/kps/compare_kernel.cu
  kernels/kps/logical_kernel.cu
  kernels/kps/reduce_kernel.cu
  kernels/cuda_kernels/abs_kernel_register.cc
  kernels/cuda_kernels/activation_kernel_register.cc
  kernels/cuda_kernels/activation_grad_kernel_register.cc
  kernels/cuda_kernels/adamw_kernel_register.cc
  kernels/cuda_kernels/addmm_kernel.cu
  kernels/cuda_kernels/amp_kernel_register.cc
  kernels/cuda_kernels/arange_kernel_register.cc
  kernels/cuda_kernels/assign_kernel.cc
  kernels/cuda_kernels/elementwise_grad_kernel_register.cc
  kernels/cuda_kernels/embedding_kernel_register.cc
  kernels/cuda_kernels/elementwise_kernel.cu
  kernels/cuda_kernels/c_identity_kernel_register.cc
  kernels/cuda_kernels/c_softmax_with_cross_entropy_kernel.cu
  kernels/cuda_kernels/cast_kernel_register.cc
  kernels/cuda_kernels/c_embedding_grad_kernel.cu
  kernels/cuda_kernels/c_embedding_kernel.cu
  kernels/cuda_kernels/clip_grad_kernel_register.cc
  kernels/cuda_kernels/clip_kernel_register.cc
  kernels/cuda_kernels/concat_kernel_register.cc
  kernels/cuda_kernels/concat_grad_kernel_register.cc
  kernels/cuda_kernels/contiguous_kernel_register.cc
  kernels/cuda_kernels/cross_entropy_kernel.cu
  kernels/cuda_kernels/cross_entropy_grad_kernel.cu
  kernels/cuda_kernels/cum_kernel_register.cc
  kernels/cuda_kernels/cum_grad_kernel_register.cc
  kernels/cuda_kernels/einsum_kernel_register.cc
  kernels/cuda_kernels/embedding_kernel_register.cc
  kernels/cuda_kernels/embedding_grad_kernel_register.cc
  kernels/cuda_kernels/expand_kernel_register.cc
  kernels/cuda_kernels/flash_attn_grad_kernel.cu
  kernels/cuda_kernels/flash_attn_kernel.cu
  kernels/cuda_kernels/fill_kernel_register.cc
  kernels/cuda_kernels/full_kernel_register.cc
  kernels/cuda_kernels/fused_rope_grad_kernel.cu
  kernels/cuda_kernels/fused_rope_kernel.cu
  kernels/cuda_kernels/gather_grad_kernel_register.cc
  kernels/cuda_kernels/gather_kernel_register.cc
  kernels/cuda_kernels/gather_nd_kernel_register.cc
  kernels/cuda_kernels/gather_nd_grad_kernel_register.cc
  kernels/cuda_kernels/gaussian_kernel_register.cc
  kernels/cuda_kernels/layer_norm_grad_kernel.cu
  kernels/cuda_kernels/layer_norm_kernel.cu
  kernels/cuda_kernels/logsumexp_kernel_register.cc
  kernels/cuda_kernels/matmul_kernel.cu
  kernels/cuda_kernels/matmul_grad_kernel.cu
  kernels/cuda_kernels/multinomial_kernel_register.cc
  kernels/cuda_kernels/nonzero_kernel_register.cc
  kernels/cuda_kernels/numel_kernel_register.cc
  kernels/cuda_kernels/one_hot_kernel_register.cc
  kernels/cuda_kernels/put_along_axis_kernel_register.cc
  kernels/cuda_kernels/reduce_kernel.cu
  kernels/cuda_kernels/rms_norm_grad_kernel_register.cc
  kernels/cuda_kernels/rms_norm_kernel_register.cc
  kernels/cuda_kernels/scale_kernel_register.cc
  kernels/cuda_kernels/slice_kernel_register.cc
  kernels/cuda_kernels/slice_grad_kernel_register.cc
  kernels/cuda_kernels/softmax_kernel_register.cc
  kernels/cuda_kernels/split_kernel_register.cc
  kernels/cuda_kernels/squared_l2_norm_kernel_register.cc
  kernels/cuda_kernels/stack_kernel_register.cc
  kernels/cuda_kernels/swiglu_kernel_register.cc
  kernels/cuda_kernels/swiglu_grad_kernel_register.cc
  kernels/cuda_kernels/take_along_axis_kernel_register.cc
  kernels/cuda_kernels/tile_grad_kernel_register.cc
  kernels/cuda_kernels/tile_kernel_register.cc
  kernels/cuda_kernels/top_k_kernel.cu
  kernels/cuda_kernels/transpose_kernel_register.cc
  kernels/cuda_kernels/transpose_grad_kernel_register.cc
  kernels/cuda_kernels/tril_triu_kernel_register.cc
  kernels/cuda_kernels/unbind_kernel_register.cc
  kernels/cuda_kernels/uniform_kernel_register.cc
  kernels/cuda_kernels/where_kernel_register.cc
  kernels/cuda_kernels/index_put_kernel_register.cc
)

set(CUSTOM_DEVICE_SRCS ${CUDA_SRCS} ${CC_SRCS})

set_source_files_properties(${CUSTOM_DEVICE_SRCS} PROPERTIES LANGUAGE CUDA)

add_library(${TARGET_NAME} SHARED ${CUSTOM_DEVICE_SRCS})
add_dependencies(${TARGET_NAME} external_deps)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PADDLE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/kernels
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME} PRIVATE
    glog
    ${PADDLE_CORE_LIB}
    cudart
    eigen3
    gflags
    protobuf
)

include_directories(BEFORE ${PADDLE_SOURCE_DIR})

target_compile_definitions(${TARGET_NAME} PUBLIC 
  PADDLE_WITH_CUDA=1 
  PADDLE_WITH_CUSTOM_DEVICE=1
  GPUContext=CustomContext
  STREAM_TYPE=cudaStream_t
  EVENT_TYPE=cudaEvent_t
  KPSContext=CustomContext
  __NVCC__=__IXCC__
)

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
