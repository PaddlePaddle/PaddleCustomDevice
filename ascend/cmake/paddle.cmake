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
# limitations under the License.

find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)

if(DEFINED ENV{PADDLE_CUSTOM_PATH})
    set(PADDLE_DIR $ENV{PADDLE_CUSTOM_PATH})
else()
    set(PADDLE_DIR ${Python_SITEARCH}/paddle)
endif()

if(NOT EXISTS ${PADDLE_DIR})
    message (FATAL_ERROR "NO Installed Paddle Found in ${PADDLE_DIR}")
endif()

set(PADDLE_INC_DIR     "${PADDLE_DIR}/include/")
set(PADDLE_LIB_DIR     "${PADDLE_DIR}/fluid/")

INCLUDE_DIRECTORIES(${PADDLE_INC_DIR})

find_file(CORE_AVX_FOUND core_avx.so ${PADDLE_LIB_DIR})
if (CORE_AVX_FOUND)
    set(paddle_lib  "${PADDLE_LIB_DIR}/core_avx.so")
else()
    set(paddle_lib  "${PADDLE_LIB_DIR}/core_noavx.so")
endif()

ADD_LIBRARY(core_lib SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET core_lib PROPERTY IMPORTED_LOCATION ${paddle_lib})

add_custom_target(paddle_install DEPENDS core_lib)
