# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.
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

project(paddle-musa CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(MUSA_PATH
    "/usr/local/musa"
    CACHE PATH "Path to which musa has been installed")
list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")
find_package(MUSA REQUIRED)

set(PLUGIN_NAME "paddle-musa")
set(PLUGIN_VERSION "1.0.0")
set(PADDLE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle/")
set(THIRD_PARTY_PATH
    "${PADDLE_SOURCE_DIR}/build/third_party"
    CACHE PATH "Third party libraries directory.")

option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_MKL "compile with mkl support" ON)
option(WITH_COVERAGE "Compile PaddlePaddle with code coverage" OFF)
option(WITH_MUSA "Compile PaddlePaddle with MUSA" ON)
option(WITH_MCCL "Compile PaddlePaddle with MCCL support" ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

if(NOT WITH_TESTING)
  set(NO_PADDLE_SUBMODULE ON)
endif()

if(NOT WITH_MUSA AND WITH_MCCL)
  message(
    WARNING "Disable MCCL when compiling without MUSA. Force WITH_MCCL=OFF.")
  set(WITH_MCCL
      OFF
      CACHE STRING "Disable MCCL when compiling without MUSA" FORCE)
endif()

# build shared library
include(utils)
include(paddle)
include(external/eigen)
include(version)
include(generic)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(
  ${PADDLE_INC_DIR}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/kernels
  ${CMAKE_SOURCE_DIR}/kernels/paddle_kernels
  ${CMAKE_SOURCE_DIR}/hack/cuda_hack
  ${CMAKE_SOURCE_DIR}/hack/thrust_hack
  ${CMAKE_SOURCE_DIR}/hack/paddle_backend_dyload_hack
  ${CMAKE_SOURCE_DIR}/hack/paddle_platform_dyload_hack)
include_directories(${PADDLE_SOURCE_DIR})
file(
  GLOB
  PADDLE_KERNEL_SRCS
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy2_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy2_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/activation_kernel.cc
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamw_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adamw_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamax_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_kernel.cu)
file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/runtime/*.cc kernels/*.cc kernels/paddle_kernels/*.cc
  # kernels/paddle_kernels/*.mu
  hack/paddle_backend_dyload_hack/*.cc hack/paddle_platform_dyload_hack/*.cc)
set(CUSTOM_DEVICE_SRCS ${PLUGIN_SRCS} ${PADDLE_KERNEL_SRCS})
# target_include_directories(${PLUGIN_NAME} PUBLIC
# ${CMAKE_SOURCE_DIR}/hack/paddle_backend_dyload_hack)
include(musa)
include(mudnn)
musa_add_library(${PLUGIN_NAME} SHARED ${CUSTOM_DEVICE_SRCS})
target_include_directories(
  ${PLUGIN_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/hack/cuda_hack
                        ${CMAKE_SOURCE_DIR}/hack/thrust_hack)
link_directories(${PADDLE_LIB_DIR})

# link paddle core lib
if(ON_INFER)
  target_link_directories(${PLUGIN_NAME} ${PADDLE_INFERENCE_LIB_DIR})
  target_link_libraries(${PLUGIN_NAME} paddle_inference)
else()
  target_link_libraries(${PLUGIN_NAME} ${PADDLE_CORE_LIB})
  # target_link_libraries(${PLUGIN_NAME} PRIVATE pybind)
endif()

# link musa rt mudnn
target_link_libraries(${PLUGIN_NAME} ${musart_lib} ${mudnn_lib})
# target_link_libraries(${PLUGIN_NAME} ${mudnn_lib})

# link mccl
if(WITH_MCCL)
  add_definitions("-DPADDLE_WITH_MCCL")
  include(mccl)
  target_link_libraries(${PLUGIN_NAME} ${mccl_lib})
else()
  if(WITH_MUSA)
    message(
      WARNING
        "If the environment is multi-card, the WITH_MCCL option needs to be turned on, otherwise only a single card can be used."
    )
  endif()
endif()

# link third party
include(third_party)
add_dependencies(${PLUGIN_NAME} third_party)
target_link_libraries(${PLUGIN_NAME} eigen3 gflags glog)
target_compile_definitions(
  ${PLUGIN_NAME}
  PUBLIC __NVCC__=__MUSACC__
         __CUDACC_VER_MAJOR__=99
         __CUDACC_VER_MINOR__=10000
         MARCH_TYPE=310 # TODO: the version num should be obtained automatically
                        # here
         GPUContext=CustomContext
         KPSContext=CustomContext
         STREAM_TYPE=musaStream_t
         EVENT_TYPE=musaEvent_t
         EIGEN_USE_GPU=1
         PADDLE_WITH_MUSA=1
         PADDLE_WITH_CUSTOM_DEVICE=1
         EIGEN_USE_MUSA=1)

# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
  TARGET ${PLUGIN_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating plugin directories------>>>")

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${PLUGIN_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
  enable_testing()
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()
