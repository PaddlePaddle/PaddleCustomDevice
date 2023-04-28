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
# the License.

find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter
                                                          Development)

if(DEFINED ENV{PADDLE_CUSTOM_PATH})
  set(PADDLE_DIR $ENV{PADDLE_CUSTOM_PATH})
else()
  set(PADDLE_DIR ${Python_SITEARCH}/paddle)
endif()

if(NOT EXISTS ${PADDLE_DIR})
  message(FATAL_ERROR "NO Installed Paddle Found in ${PADDLE_DIR}")
endif()

set(PADDLE_INC_DIR "${PADDLE_DIR}/include/")
set(PADDLE_LIB_DIR "${PADDLE_DIR}/fluid/")

include_directories(${PADDLE_INC_DIR})

if(EXISTS "${PADDLE_LIB_DIR}/libpaddle.so")
  set(paddle_lib_name libpaddle.so)
elseif(EXISTS "${PADDLE_LIB_DIR}/core_avx.so")
  set(paddle_lib_name core_avx.so)
else()
  set(paddle_lib_name core_noavx.so)
  message(WANRING "Cannot find core_avx.so, using core_noavx.so instead.")
endif()

find_library(PADDLE_CORE_LIB ${paddle_lib_name} PATHS ${PADDLE_LIB_DIR})
if(NOT PADDLE_CORE_LIB)
  message(FATAL "${paddle_lib_name} NOT found in ${PADDLE_LIB_DIR}")
else()
  message(STATUS "Found PADDLE_CORE_LIB: ${PADDLE_CORE_LIB}")
endif()

set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle/)
message(STATUS "PADDLE_SOURCE_DIR: ${PADDLE_SOURCE_DIR}")
