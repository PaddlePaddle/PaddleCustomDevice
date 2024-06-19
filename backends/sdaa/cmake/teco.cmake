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

include(version)

if(NOT DEFINED ENV{TECODNN_ROOT})
  message(FATAL
          "Cannot find TECODNN Library, please set TECODNN_ROOT correctly.")
endif()
set(TECODNN_PATH $ENV{TECODNN_ROOT})

find_path(
  TECODNN_INC
  NAMES "tecodnn.h"
  PATHS ${TECODNN_PATH}/include
  NO_DEFAULT_PATH)
find_library(
  TECODNN_LIB
  NAMES "tecodnn"
  PATHS ${TECODNN_PATH}/lib64
  NO_DEFAULT_PATH)

if(TECODNN_INC AND TECODNN_LIB)
  message(STATUS "Found TECODNN_INC: ${TECODNN_INC}")
  message(STATUS "Found TECODNN_LIB: ${TECODNN_LIB}")
elseif(NOT TECODNN_INC)
  message(FATAL "Cannot find tecodnn.h in ${TECODNN_PATH}/include")
elseif(NOT TECODNN_LIB)
  message(FATAL "Cannot find libtecodnn in ${TECODNN_PATH}/lib")
endif()

include_directories(${TECODNN_INC})

if(NOT DEFINED ENV{TBLAS_ROOT})
  message(FATAL
          "Cannot find Tecoblas Library, please set TBLAS_ROOT correctly.")
endif()
set(TBLAS_PATH $ENV{TBLAS_ROOT})

find_path(
  TBLAS_INC
  NAMES "tecoblas.h"
  PATHS ${TBLAS_PATH}/include
  NO_DEFAULT_PATH)
find_library(
  TBLAS_LIB
  NAMES "tecoblas"
  PATHS ${TBLAS_PATH}/lib64
  NO_DEFAULT_PATH)

if(TBLAS_INC AND TBLAS_LIB)
  message(STATUS "Found TBLAS_INC: ${TBLAS_INC}")
  message(STATUS "Found TBLAS_LIB: ${TBLAS_LIB}")
elseif(NOT TBLAS_INC)
  message(FATAL "Cannot find tecoblas.h in ${TBLAS_PATH}/include")
elseif(NOT TBLAS_LIB)
  message(FATAL "Cannot find libtecoblas in ${TBLAS_PATH}/lib")
endif()

include_directories(${TBLAS_INC})

if(NOT DEFINED ENV{TECODNN_CUSTOM_ROOT})
  message(
    FATAL
    "Cannot find tecodnn-custom Library, please set TECODNN_CUSTOM_ROOT correctly."
  )
endif()
set(TECODNN_CUSTOM_PATH $ENV{TECODNN_CUSTOM_ROOT})

find_path(
  TECODNN_CUSTOM_INC
  NAMES "tecodnn_custom.h"
  PATHS ${TECODNN_CUSTOM_PATH}/include
  NO_DEFAULT_PATH)
find_library(
  TECODNN_CUSTOM_LIB
  NAMES "tecodnn_ext"
  PATHS ${TECODNN_CUSTOM_PATH}/lib64
  NO_DEFAULT_PATH)

if(TECODNN_CUSTOM_INC AND TECODNN_CUSTOM_LIB)
  message(STATUS "Found TECODNN_CUSTOM_INC: ${TECODNN_CUSTOM_INC}")
  message(STATUS "Found TECODNN_CUSTOM_LIB: ${TECODNN_CUSTOM_LIB}")
elseif(NOT TECODNN_CUSTOM_INC)
  message(FATAL
          "Cannot find tecodnn_custom.h in ${TECODNN_CUSTOM_PATH}/include")
elseif(NOT TECODNN_CUSTOM_LIB)
  message(FATAL "Cannot find libtecodnn_ext in ${TECODNN_CUSTOM_PATH}/lib")
endif()

include_directories(${TECODNN_CUSTOM_INC})

if(NOT DEFINED ENV{SDPTI_ROOT})
  message(FATAL "Cannot find SDPTI_ROOT, please set SDPTI_ROOT correctly.")
endif()
set(SDPTI_PATH $ENV{SDPTI_ROOT})

find_path(
  SDPTI_INC
  NAMES "sdpti.h"
  PATHS ${SDPTI_PATH}/include
  NO_DEFAULT_PATH)
find_library(
  SDPTI_LIB
  NAMES "sdpti"
  PATHS ${SDPTI_PATH}/lib64
  NO_DEFAULT_PATH)

if(SDPTI_INC AND SDPTI_LIB)
  message(STATUS "Found SDPTI_INC: ${SDPTI_INC}")
  message(STATUS "Found SDPTI_LIB: ${SDPTI_LIB}")
  get_filename_component(SDPTI_LIBRARY_PATH ${SDPTI_LIB} DIRECTORY)
elseif(NOT SDPTI_INC)
  message(FATAL "Cannot find sdpti.h in ${SDPTI_PATH}/include")
elseif(NOT SDPTI_LIB)
  message(FATAL "Cannot find libsdpti in ${SDPTI_PATH}/lib")
endif()

include_directories(${SDPTI_INC})

if(NATIVE_SDAA)
  if(DEFINED ENV{SDAA_ROOT})
    set(SDAA_PATH $ENV{SDAA_ROOT})
  endif()

  find_path(
    SDAA_INC
    NAMES "sdaa_runtime.h"
    PATHS ${SDAA_PATH}/include
    NO_DEFAULT_PATH)
  find_library(
    SDAA_LIB
    NAMES "sdaart"
    PATHS ${SDAA_PATH}/lib64
    NO_DEFAULT_PATH)

  include_directories(${SDAA_INC})

  if(SDAA_INC AND SDAA_LIB)
    set(SDAA_FOUND ON)
    message(STATUS "Found SDAA_INC: ${SDAA_INC}")
    message(STATUS "Found SDAA_LIB: ${SDAA_LIB}")
  elseif(NOT SDAA_INC)
    message(FATAL "Cannot find sdaa_runtime.h in ${SDAA_PATH}/include")
  else()
    message(FATAL "Cannot find libsdaart.so in ${SDAA_PATH}/lib64")
  endif()
endif()

if(NOT DEFINED ENV{EXTEND_OP_ROOT})
  message(FATAL
          "Cannot find EXTEND_OP Library, please set EXTEND_OP_ROOT correctly.")
endif()
set(EXTEND_OP_PATH $ENV{EXTEND_OP_ROOT})

find_path(
  EXTEND_OP_INC
  NAMES "sdcops.h"
  PATHS ${EXTEND_OP_PATH}/include
  NO_DEFAULT_PATH)

find_library(
  EXTEND_OP_LIB
  NAMES "libsdcops.so"
  PATHS ${EXTEND_OP_PATH}/lib
  NO_DEFAULT_PATH)

if(EXTEND_OP_INC AND EXTEND_OP_LIB)
  message(STATUS "Found EXTEND_OP_INC: ${EXTEND_OP_INC}")
  message(STATUS "Found EXTEND_OP_LIB: ${EXTEND_OP_LIB}")
elseif(NOT EXTEND_OP_INC)
  message(FATAL "Cannot find sdcops.h in ${EXTEND_OP_PATH}/include")
elseif(NOT EXTEND_OP_LIB)
  message(FATAL "Cannot find libsdcops.so in ${EXTEND_OP_PATH}/lib")
endif()

if(NOT DEFINED ENV{TCCL_ROOT})
  message(FATAL "Cannot find TCCL Library, please set TCCL_ROOT correctly.")
endif()
set(TCCL_PATH $ENV{TCCL_ROOT})

find_path(
  TCCL_INC
  NAMES "tccl.h"
  PATHS ${TCCL_PATH}/include
  NO_DEFAULT_PATH)

find_library(
  TCCL_LIB
  NAMES "tccl"
  PATHS ${TCCL_PATH}/lib64
  NO_DEFAULT_PATH)

if(TCCL_INC AND TCCL_LIB)
  message(STATUS "Found TCCL_INC: ${TCCL_INC}")
  message(STATUS "Found TCCL_LIB: ${TCCL_LIB}")
elseif(NOT TCCL_INC)
  message(FATAL "Cannot find tccl.h in ${TCCL_PATH}/include")
elseif(NOT TCCL_LIB)
  message(FATAL "Cannot find libtccl in ${TCCL_PATH}/lib64")
endif()
