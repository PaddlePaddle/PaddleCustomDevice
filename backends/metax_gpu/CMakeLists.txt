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
set(PYTHON_VERSION "3.8")
project(paddle-metax-gpu CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle/cmake")
set(CMAKE_VERBOSE_MAKEFILE ON)


include(paddle)
include(version)
include(generic)

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)

add_definitions(-DGPUContext=CustomContext)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(-DSTREAM_TYPE=cudaStream_t)
add_definitions(-DEVENT_TYPE=cudaEvent_t)

message(STATUS "CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR:${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "PADDLE INIT DONE!")
option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_GPU "Compile PaddlePaddle with METAX_GPU" ON)

enable_language(CUDA)
include(../../Paddle/cmake/cuda.cmake)

message(STATUS "CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")

set(PLUGIN_NAME "paddle-metax-gpu")
set(TEST_TARGET "paddle-metax-gpu-test")
set(PLUGIN_VERSION "0.0.1")

set(RUNTIME_SRCS runtime/runtime.cc)

include_directories(${CMAKE_SOURCE_DIR} ${PADDLE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/kernels)

add_definitions(-std=c++14)

message(STATUS "PADDLE_SOURCE_DIR = ${PADDLE_SOURCE_DIR}")
set(THIRD_PARTY_PATH "${PADDLE_SOURCE_DIR}/build/third_party" CACHE PATH "Third party libraries directory.")
message(STATUS "THIRD_PARTY_PATH = ${THIRD_PARTY_PATH}")
include_directories(${CMAKE_SOURCE_DIR} ${PADDLE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/kernels)

file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.cc)
list(APPEND PLUGIN_SRCS runtime/runtime.cc)

include_directories(${PADDLE_SOURCE_DIR})

set(PP_3RD ${PADDLE_SOURCE_DIR}/cmake/external/)
set(PP_gflags ${PADDLE_SOURCE_DIR}/cmake/external/gflags.cmake)
set(PP_glog ${PADDLE_SOURCE_DIR}/cmake/external/glog.cmake)
set(PP_eigen ${PADDLE_SOURCE_DIR}/cmake/external/eigen.cmake)
include(${PP_gflags})
include(${PP_glog})
include(${PP_eigen})

message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
set(CU_FILE_FOLDER ${PADDLE_SOURCE_DIR}/Paddle/paddle/phi/kernels/gpu/)
file(
  GLOB CUDA_SRCS
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_add_kernel.cu)
  list(APPEND CUDA_SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eye_kernel.cu)
  # list(APPEND CUDA_SRCS ${PADDLE_SOURCE_DIR}/paddle/common/flags.cc)

file(
    GLOB CC_SRCS
    ${CMAKE_SOURCE_DIR}/kernels/cuda_kernels/index_add_kernel_register.cc)
    list(APPEND CC_SRCS ${CMAKE_SOURCE_DIR}/kernels/cuda_kernels/eye_kernel_register.cc)

message(STATUS "CUDA_SRCS = ${CUDA_SRCS}")
message(STATUS "CC_SRCS = ${CC_SRCS}")

set(CUSTOM_GPU_SRCS ${CUDA_SRCS} ${CC_SRCS} ${RUNTIME_SRCS})

message(STATUS "PADDLE_CORE_LIB = ${PADDLE_CORE_LIB}")
include_directories(/opt/conda/lib/python3.8/site-packages/paddle/include)
link_directories(/opt/conda/lib/python3.8/site-packages/paddle/libs)
add_library(${TEST_TARGET} SHARED ${CUSTOM_GPU_SRCS})
target_link_libraries(${TEST_TARGET} PRIVATE
    glog
    ${PADDLE_CORE_LIB}
    /opt/conda/lib/python3.8/site-packages/paddle/libs/libcommon.so
    /opt/conda/lib/python3.8/site-packages/paddle/libs/libphi_core.so
)
# set(CMAKE_INSTALL_RPATH "$ORIGIN:/opt/conda/lib/python3.8/site-packages/paddle/libs")

include_directories(BEFORE ${PADDLE_SOURCE_DIR})

target_compile_definitions(${TEST_TARGET} PUBLIC PADDLE_WITH_CUDA=1 
    PADDLE_WITH_CUSTOM_DEVICE=1
    STREAM_TYPE=cudaStream_t )

set(PREPROCESSED_FILES)
foreach(cuda_src ${CUDA_SRCS})
  get_filename_component(cuda_name ${cuda_src} NAME_WE)
  set(preprocessed_file ${cuda_name}_preprocessed.cc)
  add_custom_command(
    OUTPUT ${preprocessed_file}
    COMMAND nvcc -E ${cuda_src}
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
      -I${PADDLE_SOURCE_DIR}/build/third_party/install/glog/include
      -I${PADDLE_SOURCE_DIR}/build/third_party/install/gflags
      -o ${CMAKE_CURRENT_BINARY_DIR}/${preprocessed_file}
      DEPENDS ${PHI_DEPS})
  message("------------------- ${PHI_DEPS}")
  list(APPEND PREPROCESSED_FILES ${preprocessed_file})
endforeach()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}")


add_custom_target(mypreprocess
    DEPENDS ${PREPROCESSED_FILES}
)
add_dependencies(${TEST_TARGET} mypreprocess)


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

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${TEST_TARGET}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)




