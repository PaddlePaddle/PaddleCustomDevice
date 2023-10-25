# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

project(paddle-custom-gcu CXX C)
set(CUSTOM_GCU_NAME "paddle-custom-gcu")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_DEBUG CMAKE_MODULE_PATH is: ${CMAKE_MODULE_PATH}")

set(TOPS_RELEASE_VERSION 2.4.1)
if((NOT DEFINED PACKAGE_VERSION)
   OR ("${PACKAGE_VERSION}" STREQUAL "")
   OR ("${PACKAGE_VERSION}" STREQUAL "123.456"))
  # get Paddle-custom git hash
  execute_process(
    COMMAND git log -1 --abbrev=7 --format=%h
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "git hash: ${GIT_HASH}")
  set(GIT_HASH ".${GIT_HASH}")
else()
  set(GIT_HASH "")
  set(TOPS_RELEASE_VERSION ${PACKAGE_VERSION})
endif()
set(PACKAGE_VERSION ${GIT_HASH}-${TOPS_RELEASE_VERSION})

message(STATUS "tops release version: ${TOPS_RELEASE_VERSION}")
message(STATUS "package version: ${PACKAGE_VERSION}")

include(paddle)
include(version)
include(generic)
include(external/gcu)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(/opt/tops/include)

set(OUTPUT_PADDLE_PACKAGE_VERSION ${PADDLE_VERSION}${PACKAGE_VERSION})
option(WITH_MKLDNN "compile with MKLDNN support" ON)
option(WITH_KERNELS "compile with custom kernels" ON)
option(WITH_TESTING "compile with unit testing" OFF)
option(WITH_ARM "compile with arm support" OFF)
option(ON_INFER "compile with inference c++ lib" OFF)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")
# Note: _GLIBCXX_USE_CXX11_ABI should be same with libpaddle.so, default is 1
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

# custom runtime
set(CUSTOM_GCU_SRCS runtime/runtime.cc)
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)
if(WITH_ARM)
  add_definitions(-DPADDLE_WITH_ARM)
endif()

# custom kernels
if(WITH_KERNELS)
  add_compile_definitions(HLIR_BUILDER_ABI_COMPATIABLE)
  file(
    GLOB_RECURSE CUSTOM_KERNEL_SRCS
    RELATIVE ${CMAKE_SOURCE_DIR}
    kernels/*.cc backend/*.cc common/*.cc)
  list(APPEND CUSTOM_GCU_SRCS ${CUSTOM_KERNEL_SRCS})
endif()

if(ON_INFER)
  link_directories(${PADDLE_INFERENCE_LIB_DIR})
endif()

# build shared library
add_library(${CUSTOM_GCU_NAME} SHARED ${CUSTOM_GCU_SRCS})
target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE ${GCU_LIBS})
if(TOPS_MODULE_SOURCE_DIR)
  target_include_directories(${CUSTOM_GCU_NAME}
                             PRIVATE ${TOPS_MODULE_SOURCE_DIR}/usr/include)
  target_include_directories(${CUSTOM_GCU_NAME}
                             PRIVATE ${TOPS_MODULE_SOURCE_DIR}/opt/tops/include)
endif()

# link paddle shared library
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party.cmake)
add_dependencies(${CUSTOM_GCU_NAME} third_party)
target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE glog gflags)
if(ON_INFER)
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE paddle_inference)
  target_compile_definitions(${CUSTOM_GCU_NAME} PRIVATE PADDLE_ON_INFERENCE)
else()
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE ${PADDLE_CORE_LIB})
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE pybind)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION} REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIR})
endif()

# testing
if(WITH_TESTING)
  enable_testing()
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../python")
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()

# get git commit id
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Git commit id is: ${GIT_HASH}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

message(STATUS "CMAKE_DEBUG Git commit id is: ${GIT_HASH}")
message(STATUS "CMAKE_DEBUG CUSTOM_GCU_SRCS is: ${CUSTOM_GCU_SRCS}")
message(STATUS "CMAKE_DEBUG GCU_LIBS is: ${GCU_LIBS}")

# packing wheel package
add_custom_command(
  TARGET ${CUSTOM_GCU_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${CUSTOM_GCU_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating custom device directories------>>>")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${CUSTOM_GCU_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
