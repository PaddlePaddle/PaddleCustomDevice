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

if(NOT ENV{NEUWARE_HOME})
  set(NEUWARE_HOME "/usr/local/neuware")
else()
  set(NEUWARE_HOME $ENV{NEUWARE_HOME})
endif()
message(STATUS "NEUWARE_HOME: " ${NEUWARE_HOME})

set(NEUWARE_INCLUDE_DIR ${NEUWARE_HOME}/include)
set(NEUWARE_LIB_DIR ${NEUWARE_HOME}/lib64)

include_directories(${NEUWARE_INCLUDE_DIR})

set(CNNL_LIB ${NEUWARE_LIB_DIR}/libcnnl.so)
set(CNRT_LIB ${NEUWARE_LIB_DIR}/libcnrt.so)
set(CNPAPI_LIB ${NEUWARE_LIB_DIR}/libcnpapi.so)
set(CNCL_LIB ${NEUWARE_LIB_DIR}/libcncl.so)

set(NEUWARE_LIBS ${CNNL_LIB} ${CNRT_LIB} ${CNPAPI_LIB} ${CNCL_LIB})

