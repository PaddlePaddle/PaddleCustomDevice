# Copyright(c) 2023 PaddlePaddle Authors.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0(the "License"); you may not use
# this file except in compliance with the License.You may obtain a copy of the
# License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.See the
# License for the specific language governing permissions and limitations under
# the License

cmake_minimum_required(VERSION 3.10)

project(paddle-mps CXX C)

set(SIGN_IDENTITY
    ""
    CACHE STRING "Code signing identity for the dylib")

if(SIGN_IDENTITY STREQUAL "")
  message(FATAL_ERROR "SIGN_IDENTITY must be set")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ${SIGN_IDENTITY})
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "YES")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_TESTING "compile with unit testing" OFF)
option(ON_INFER "compile with inference c++ lib" OFF)

set(PLUGIN_NAME "paddle-mps")
set(PLUGIN_VERSION "0.0.1")

include(paddle)

include_directories(${PADDLE_INC_DIR} ${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/kernels ${CMAKE_SOURCE_DIR}/runtime)
link_directories(${PADDLE_LIB_DIR})

file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.mm ${CMAKE_SOURCE_DIR} kernels/*.cc
  ${CMAKE_SOURCE_DIR}/runtime/*.mm)
list(APPEND PLUGIN_SRCS runtime/runtime.cc)

# build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
if(ON_INFER)
  target_link_directories(${PLUGIN_NAME} PRIVATE ${PADDLE_INFERENCE_LIB_DIR})
  target_link_libraries(${PLUGIN_NAME} PRIVATE paddle_inference)
else()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})
endif()

find_library(FOUNDATION_LIBRARY Foundation)
find_library(METAL_LIBRARY Metal REQUIRED)
find_library(MPS_LIBRARY MetalPerformanceShaders REQUIRED)
find_library(MPS_GRAPH_LIBRARY MetalPerformanceShadersGraph REQUIRED)
target_link_libraries(
  ${PLUGIN_NAME} PRIVATE ${METAL_LIBRARY} ${MPS_LIBRARY} ${FOUNDATION_LIBRARY}
                         ${MPS_GRAPH_LIBRARY})

# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
  TARGET ${PLUGIN_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.dylib
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
  COMMAND
    install_name_tool -change @loader_path/../libs/ ${PADDLE_CORE_LIB}
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/lib${PLUGIN_NAME}.dylib
  COMMENT "Creating plugin dirrectories------>>>")

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
