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

#NOTE: Logic is from
# https://github.com/mindspore-ai/graphengine/blob/master/CMakeLists.txt
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_DIR /usr/local/Ascend)
endif()

set(ASCEND_CL_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/lib64)

set(ascend_hccl_lib ${ASCEND_CL_DIR}/libhccl.so)
set(ascendcl_lib ${ASCEND_CL_DIR}/libascendcl.so)
set(acl_op_compiler_lib ${ASCEND_CL_DIR}/libacl_op_compiler.so)
set(FWKACLLIB_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include)
set(ACLLIB_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/acllib/include)
set(OPS_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/opp/built-in/op_proto/inc)
set(COMPILER_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/compiler/include/ge)

# link_directories(${ASCEND_CL_DIR} ${ASCEND_DIR}/ascend-toolkit/latest/opp/op_proto/built-in/ ${ASCEND_DIR}/ascend-toolkit/latest/runtime/lib64)

message(STATUS "FWKACLLIB_INC_DIR ${FWKACLLIB_INC_DIR}")
message(STATUS "ASCEND_CL_DIR ${ASCEND_CL_DIR}")
message(STATUS "OPS_INC_DIR ${OPS_INC_DIR}")
message(STATUS "COMPILER_INC_DIR ${COMPILER_INC_DIR}")
INCLUDE_DIRECTORIES(${FWKACLLIB_INC_DIR})
INCLUDE_DIRECTORIES(${ACLLIB_INC_DIR})
INCLUDE_DIRECTORIES(${OPS_INC_DIR})
INCLUDE_DIRECTORIES(${COMPILER_INC_DIR})

ADD_LIBRARY(ascendcl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascendcl PROPERTY IMPORTED_LOCATION ${ascendcl_lib})

ADD_LIBRARY(ascend_hccl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascend_hccl PROPERTY IMPORTED_LOCATION ${ascend_hccl_lib})

ADD_LIBRARY(acl_op_compiler SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET acl_op_compiler PROPERTY IMPORTED_LOCATION ${acl_op_compiler_lib})
add_custom_target(ascend_cl DEPENDS ascendcl acl_op_compiler)

macro(find_ascend_toolkit_version ascend_toolkit_version_info) 
    file(READ ${ascend_toolkit_version_info} ASCEND_TOOLKIT_VERSION_CONTENTS)
    string(REGEX MATCH "version=([0-9]+\.[0-9]+\.(RC)?[0-9]*)" ASCEND_TOOLKIT_VERSION "${ASCEND_TOOLKIT_VERSION_CONTENTS}")
    string(REGEX REPLACE "version=([0-9]+\.[0-9]+\.(RC)?[0-9]*)" "\\1" ASCEND_TOOLKIT_VERSION "${ASCEND_TOOLKIT_VERSION}")
    string(REGEX REPLACE "[A-Z]|[a-z|\.]" "" CANN_VERSION ${ASCEND_TOOLKIT_VERSION})
    string(SUBSTRING "${CANN_VERSION}000" 0 6 CANN_VERSION)
    add_definitions("-DCANN_VERSION_CODE=${CANN_VERSION}")
    if(NOT ASCEND_TOOLKIT_VERSION)
        set(ASCEND_TOOLKIT_VERSION "???")
    else()
        message(STATUS "Current Ascend Toolkit version is ${ASCEND_TOOLKIT_VERSION}")
    endif()
endmacro()

macro(find_ascend_driver_version ascend_driver_version_info) 
    file(READ ${ascend_driver_version_info} ASCEND_DRIVER_VERSION_CONTENTS)
    string(REGEX MATCH "Version=([0-9]+\.[0-9]+\.[0-9]+)" ASCEND_DRIVER_VERSION "${ASCEND_DRIVER_VERSION_CONTENTS}")
    string(REGEX REPLACE "Version=([0-9]+\.[0-9]+\.[0-9]+)" "\\1" ASCEND_DRIVER_VERSION "${ASCEND_DRIVER_VERSION}")
    if(NOT ASCEND_DRIVER_VERSION)
        set(ASCEND_DRIVER_VERSION "???")
    else()
        message(STATUS "Current Ascend Driver version is ${ASCEND_DRIVER_VERSION}")
    endif()
endmacro()

if (WITH_ARM)
  set(ASCEND_TOOLKIT_DIR ${ASCEND_DIR}/ascend-toolkit/latest/arm64-linux)
else()
  set(ASCEND_TOOLKIT_DIR ${ASCEND_DIR}/ascend-toolkit/latest/x86_64-linux)
endif()

find_ascend_toolkit_version(${ASCEND_TOOLKIT_DIR}/ascend_toolkit_install.info)
find_ascend_driver_version(${ASCEND_DIR}/driver/version.info)
