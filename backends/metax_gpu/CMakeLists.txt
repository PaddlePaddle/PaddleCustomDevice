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

set(PLUGIN_VERSION "0.0.1")
set(TARGET_NAME ${PROJ_NAME})

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PY_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
message(STATUS "Python version detected: ${PY_VERSION}")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(paddle)
include(version)
include(generic)

list(APPEND CMAKE_MODULE_PATH "${PADDLE_SOURCE_DIR}/cmake"
     "${PADDLE_SOURCE_DIR}/cmake/external")

enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(-std=c++17)

option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_GPU "Compile PaddlePaddle with METAX_GPU" ON)

set(THIRD_PARTY_PATH
    "${PADDLE_SOURCE_DIR}/build/third_party"
    CACHE PATH "Third party libraries directory.")

include(cuda)
include(gflags)
include(glog)
include(eigen)

file(
  GLOB
  CUDA_SRCS
  ${PADDLE_SOURCE_DIR}/paddle/phi/backends/gpu/cuda/cuda_info.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math_function.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eye_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bincount_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_kernel.cu)

file(
  GLOB_RECURSE CC_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  runtime/runtime.cc kernels/*.cc kernels/*.cu kernels/cuda_kernels/*.cc
  kernels/cuda_kernels/*.cu)

set(CUSTOM_DEVICE_SRCS ${CUDA_SRCS} ${CC_SRCS})

set_source_files_properties(${CUSTOM_DEVICE_SRCS} PROPERTIES LANGUAGE CUDA)

add_library(${TARGET_NAME} SHARED ${CUSTOM_DEVICE_SRCS})

target_include_directories(
  ${TARGET_NAME} PRIVATE ${PADDLE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
                         ${CMAKE_SOURCE_DIR}/kernels ${CUDA_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} glog eigen3 gflags ${PADDLE_CORE_LIB})

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
