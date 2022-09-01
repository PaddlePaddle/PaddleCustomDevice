# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.10)

project(paddle-custom-intel-gpu CXX C)

##########################################


if(NOT DEFINED ONEAPI_PATH)
set(ONEAPI_PATH "$ENV{HOME}/intel/oneapi")
endif()

if(NOT EXISTS "${ONEAPI_PATH}")
message(FATAL_ERROR "Can't find openapi directory ${ONEAPI_PATH}")
endif()

message(STATUS "OneAPI DIR : ${ONEAPI_PATH}")

if(NOT DEFINED ONEAPI_COMPILER_DIR)
set(ONEAPI_COMPILER_DIR "compiler/2022.1.0")
endif()

if(NOT EXISTS "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}")
message(FATAL_ERROR "Can't find directory ${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}")
endif()

if(NOT DEFINED ENV{CONDA_PREFIX_1})
message(FATAL_ERROR "Please install python miniconda ")
endif()

if(NOT DEFINED CND_INSTALL_DIR)
    set(CND_INSTALL_DIR "$ENV{CONDA_PREFIX_1}")
endif()

if(NOT EXISTS "${CND_INSTALL_DIR}")
 message(FATAL_ERROR "Can't find conda directory ${CND_INSTALL_DIR}")
endif()

message(STATUS "Conda DIR : ${CND_INSTALL_DIR}")

if(NOT DEFINED CND_ENV)
set(CND_ENV "$ENV{CONDA_DEFAULT_ENV}")
endif()

message(STATUS "CONDA default env => ${CND_ENV}")

set(CND_PYTHON "${CND_INSTALL_DIR}/envs/${CND_ENV}/lib/python3.9")

if(NOT EXISTS "${CND_PYTHON}")
   message(FATAL_ERROR "Can't find conda python directory ${CND_PYTHON}" )
endif()



#############################

set(DPCPP_COMPIER_PATH "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/bin/dpcpp")
set(ONEAPI_SYCL_INCLUDE "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/include/sycl/")
set(ONEAPI_SYCL_LIBDIR "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/lib/")
set(CMAKE_CXX_COMPILER "${DPCPP_COMPIER_PATH}")
##########################################


set(ONEDNN_INC "${PADDLE_BUILD_DIR}/third_party/install/mkldnn/include")
set(ONEDNN_LIB "${PADDLE_BUILD_DIR}/third_party/install/mkldnn/lib")
##########################################
set(PADDLE_PLUGIN_DIR  "${CND_PYTHON}/site-packages/paddle-plugins/")
set(PADDLE_INC_DIR     "${CND_PYTHON}/site-packages/paddle/include/")
set(PADDLE_LIB_DIR     "${CND_PYTHON}/site-packages/paddle/fluid/")


include_directories( ${ONEAPI_SYCL_INCLUDE} ${ONEDNN_INC})

###########################################



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_TESTING    "compile with unit testing"        ON)

set(PLUGIN_NAME        "paddle-custom-intel-gpu")
set(PLUGIN_VERSION     "0.0.1")

include(paddle)

include_directories(${PADDLE_INC_DIR} ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/kernels)
link_directories(${PADDLE_LIB_DIR} ${ONEDNN_LIB} ${ONEAPI_SYCL_LIBDIR})

add_definitions(-std=c++14)

file(GLOB_RECURSE PLUGIN_SRCS RELATIVE ${CMAKE_SOURCE_DIR} kernels/*.cc)
list(APPEND PLUGIN_SRCS runtime/runtime.cc)

# build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})  # special name

if( "${CMAKE_CXX_COMPILER}" STREQUAL "${DPCPP_COMPIER_PATH}" )
        add_definitions(-DUSEDPCPP)
        set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_STANDARD 17)
endif()


# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

if (WITH_TESTING)
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
enable_testing()
add_subdirectory(tests)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(python_tests ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()
