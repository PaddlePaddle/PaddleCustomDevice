# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
set(PYTHON_VERSION "3.8.16")
project(paddle-custom-gpu CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -H")



include(paddle)
include(version)
include(generic)

# 在CMakeLists.txt中添加PADDLE_WITH_CUDA宏定义
# add_definitions(-DPADDLE_WITH_CUDA)
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)

# # TODO(qiil93): avoid compile error, to be removed
# add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

# add_definitions(-DPADDLE_WITH_GPGPU)
# add_definitions(-DMOVE_TO_GPGPU)
add_definitions(-DPADDLE_CUDA_FP16)
add_definitions(-DGPUContext=CustomContext)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

message(STATUS "CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message(STATUS "PADDLE INIT DONE!")
option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
# option(WITH_GPU "Compile PaddlePaddle with NVIDIA GPU" ON)

enable_language(CUDA)
include(../../Paddle/cmake/cuda.cmake)

message(STATUS "CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")

set(PLUGIN_NAME "paddle-custom-gpu")
set(TEST_TARGET "paddle-custom-gpu-test")
set(PLUGIN_VERSION "0.0.1")

if(NOT WITH_TESTING)
  set(NO_PADDLE_SUBMODULE ON)
endif()
# include(paddle)

set(RUNTIME_SRCS runtime/runtime.cc)

message(STATUS "PADDLE_SOURCE_DIR = ${PADDLE_SOURCE_DIR}")

include_directories(${CMAKE_SOURCE_DIR} ${PADDLE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/kernels)
# link_directories(${PADDLE_LIB_DIR})

add_definitions(-std=c++14)

file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.cc)
list(APPEND PLUGIN_SRCS runtime/runtime.cc)
#################################################################################
include_directories(${PADDLE_SOURCE_DIR})

set(PP_3RD ${PADDLE_SOURCE_DIR}/cmake/external/)
set(PP_gflags ${PADDLE_SOURCE_DIR}/cmake/external/gflags.cmake)
set(PP_glog ${PADDLE_SOURCE_DIR}/cmake/external/glog.cmake)
set(PP_eigen ${PADDLE_SOURCE_DIR}/cmake/external/eigen.cmake)
include(${PP_gflags})
include(${PP_glog})
include(${PP_eigen})


set(CU_FILE_FOLDER ${PADDLE_SOURCE_DIR}/Paddle/paddle/phi/kernels/gpu/)
file(
  GLOB TEST_SRCS
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_add_kernel.cu)
# list(APPEND TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_grad_kernel.cu)
# list(APPEND TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_kernel.cu)
# list(APPEND TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/accuracy_check_kernel.cu)
# list(APPEND TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/accuracy_kernel.cu)
# list(APPEND TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_grad_kernel.cu)
# list(APPEND TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu) #blas

# list(REMOVE_ITEM TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu)
# list(REMOVE_ITEM TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adagrad_kernel.cu)

set(TEST_SRCS ${FILE_NAME})
set(TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_grad_kernel.cu)
set(TEST_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/arg_min_max_kernel.cu) #ok
# set(TEST_SRCS /ssd3/mayongqiang/cpu/PaddleCustomDevice/Paddle/paddle/phi/kernels/gpu/elementwise_grad_kernel.cu)

message(STATUS "TEST_SRCS = ${TEST_SRCS}")


# set(CPU_RUN runtime/runtime.cc)
set(TEST_SRCS "${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_add_kernel.cu")
# set_source_files_properties(/ssd3/mayongqiang/cpu/PaddleCustomDevice/Paddle/paddle/phi/kernels/gpu/index_add_kernel.cu PROPERTIES COMPILE_DEFINITIONS "PADDLE_WITH_CUDA")
# set_source_files_properties(${TEST_SRCS} PROPERTIES COMPILE_DEFINITIONS "PADDLE_WITH_CUDA")

# set(TEST_SRCS "/ssd3/mayongqiang/cpu/PaddleCustomDevice/Paddle/paddle/phi/kernels/gpudnn/conv_transpose_kernel.cu")
# list(APPEND TEST_SRCS /ssS·d3/mayongqiang/cpu/PaddleCustomDevice/Paddle/paddle/phi/kernels/gpudnn/conv_transpose_kernel.cu)
# message(STATUS "TEST_SRCS = ${TEST_SRCS}")
# list(APPEND TEST_SRCS /ssd3/mayongqiang/cpu/PaddleCustomDevice/Paddle/paddle/phi/core/dense_tensor.cc)
# list(APPEND TEST_SRCS /ssd3/mayongqiang/cpu/PaddleCustomDevice/Paddle/paddle/phi/core/dense_tensor.inl)

set(CUSTOM_GPU_SRCS ${TEST_SRCS})
list(APPEND CUSTOM_GPU_SRCS ${RUNTIME_SRCS})

add_library(${TEST_TARGET} SHARED ${CUSTOM_GPU_SRCS})
target_link_libraries(${TEST_TARGET} PRIVATE glog)
target_link_libraries(${TEST_TARGET} PRIVATE ${PADDLE_CORE_LIB})

message("include: =====", ${C_INCLUDE_PATH})
message("include: =====", ${CPLUS_INCLUDE_PATH})
include_directories(BEFORE /home/PaddleCustomDevice/Paddle)
# set_property(
#   TEST ${TEST_TARGET}
#   PROPERTY
#     ENVIRONMENT
#     FLAGS_init_allocated_mem=true
#     FLAGS_cudnn_deterministic=true
#     FLAGS_enable_pir_api=0
#     # LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_BINARY_DIR}/python/paddle/libs:${PADDLE_BINARY_DIR}/python/paddle/base
# )

target_compile_definitions(${TEST_TARGET} PUBLIC PADDLE_WITH_CUDA=1)

add_custom_command(
  OUTPUT index_add_kernel_preprocessed.cc
  COMMAND nvcc -E ${TEST_SRCS}
    -I${PADDLE_SOURCE_DIR}
    -I${POPART_DIR}/include
    -I${JEMALLOC_INCLUDE_DIR} 
    -I${UTF8PROC_INSTALL_DIR}/include
    -I${XBYAK_INC_DIR} 
    -I${XBYAK_INC_DIR}/xbyak
    -I${SOURCE_INCLUDE_DIR}
    -I${GLOG_INCLUDE_DIR}
    -I${DIRENT_INCLUDE_DIR}
    -I${PROTOBUF_INCLUDE_DIR}
    -I${POCKETFFT_INCLUDE_DIR}
    -I${GFLAGS_INCLUDE_DIR}
    -I${GLOO_INCLUDE_DIR}
    -I${YAML_INCLUDE_DIR}
    -I${PSLIB_BRPC_INC_DIR}
    -I${XXHASH_INCLUDE_DIR}
    -I${ZLIB_INCLUDE_DIR}
    -I${GTEST_INCLUDE_DIR}
    -I${PADDLE_SOURCE_DIR}/third_party/eigen3
    -I${PADDLE_SOURCE_DIR}/third_party/install/glog/include
    -I${PADDLE_SOURCE_DIR}/third_party/gflags
    -o ${CMAKE_CURRENT_BINARY_DIR}/index_add_kernel_preprocessed.cc
    DEPENDS ${PHI_DEPS})
message("------------------- ${PHI_DEPS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}")


add_custom_target(mypreprocess
    DEPENDS index_add_kernel_preprocessed.cc
)
add_dependencies(${TEST_TARGET} mypreprocess)

#################################################################################

# # build shared library
# add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
# if(ON_INFER)
#   target_link_directories(${PLUGIN_NAME} PRIVATE ${PADDLE_INFERENCE_LIB_DIR})
#   target_link_libraries(${PLUGIN_NAME} PRIVATE paddle_inference)
# else()
#   target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})
# endif()




#################################################################################
# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
  
add_custom_command(
  TARGET ${TEST_TARGET}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${TEST_TARGET}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating plugin directories------>>>")
#################################################################################


# add_custom_command(
#   TARGET ${PLUGIN_NAME}
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
#   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
#   COMMAND ${CMAKE_COMMAND} -E make_directory
#           ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
#   COMMAND
#     ${CMAKE_COMMAND} -E copy_if_different
#     ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
#     ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
#   COMMENT "Creating plugin directories------>>>")





# find_package(
#   Python
#   COMPONENTS Interpreter
#   REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${TEST_TARGET}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

# if(WITH_TESTING)
#   set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../python")
#   add_subdirectory(tests)
#   add_custom_command(
#     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
#     COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
#   add_custom_target(python_tests ALL
#                     DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
# endif()



