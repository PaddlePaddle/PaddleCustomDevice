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

if(DEFINED $ENV{PADDLE_CUSTOM_PATH})
  set(PADDLE_DIR $ENV{PADDLE_CUSTOM_PATH})
else()
  set(PADDLE_DIR ${Python_SITEARCH}/paddle)
endif()

if(NOT EXISTS ${PADDLE_DIR})
  message(FATAL_ERROR "NO Installed Paddle Found in ${PADDLE_DIR}")
endif()

set(PADDLE_INC_DIR "${PADDLE_DIR}/include/")
set(PADDLE_LIB_DIR "${PADDLE_DIR}/base/")

include_directories(${PADDLE_INC_DIR})

if(EXISTS "${PADDLE_LIB_DIR}/libpaddle.so")
  set(paddle_lib_name libpaddle.so)
endif()

find_library(PADDLE_CORE_LIB ${paddle_lib_name} PATHS ${PADDLE_LIB_DIR})
if(NOT PADDLE_CORE_LIB)
  message(FATAL "${paddle_lib_name} NOT found in ${PADDLE_LIB_DIR}")
else()
  message(STATUS "Found PADDLE_CORE_LIB: ${PADDLE_CORE_LIB}")
endif()

function(get_paddle_info OUTPUT_COMMIT OUTPUT_VERSION OUTPUT_PATH)
  execute_process(
    COMMAND python -c "import os; os.environ['CUSTOM_DEVICE_ROOT'] = '';\
                         import paddle; print(paddle.version.commit)"
    OUTPUT_VARIABLE PADDLE_COMMIT_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND python -c "import os; os.environ['CUSTOM_DEVICE_ROOT'] = '';\
                          import paddle; print(paddle.version.full_version)"
    OUTPUT_VARIABLE PADDLE_FULL_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND python -c "import os; os.environ['CUSTOM_DEVICE_ROOT'] = '';\
                          import paddle; print(paddle.__path__[0])"
    OUTPUT_VARIABLE PADDLE_BUILD_ENV_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${OUTPUT_COMMIT}
      "${PADDLE_COMMIT_ID}"
      PARENT_SCOPE)
  set(${OUTPUT_VERSION}
      "${PADDLE_FULL_VERSION}"
      PARENT_SCOPE)
  set(${OUTPUT_PATH}
      "${PADDLE_BUILD_ENV_PATH}"
      PARENT_SCOPE)
endfunction()
