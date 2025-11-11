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

include(ExternalProject)

set(ONEDNN_GPU "extern_onednn_gpu")

set(ONEDNN_GPU_PREFIX_DIR ${THIRD_PARTY_PATH}/onednn_gpu)
set(ONEDNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/onednn_gpu)

message(STATUS "${ONEDNN_GPU} install dir : ${ONEDNN_INSTALL_DIR}")

ExternalProject_Add(
  ${ONEDNN_GPU}
  GIT_SHALLOW TRUE
  GIT_REPOSITORY "${GIT_URL}/oneapi-src/oneDNN.git"
  GIT_TAG rls-v2.7
  GIT_PROGRESS TRUE
  PREFIX ${ONEDNN_GPU_PREFIX_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=icx
             -DCMAKE_C_COMPILER=icx
             -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DDNNL_BUILD_TESTS=OFF
             -DDNNL_BUILD_EXAMPLES=OFF
             -DDNNL_GPU_RUNTIME=DPCPP)

include_directories(${ONEDNN_INSTALL_DIR}/include)
list(APPEND third_party_deps ${ONEDNN_GPU})
