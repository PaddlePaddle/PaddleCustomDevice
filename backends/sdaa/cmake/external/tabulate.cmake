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
# the License.

include(ExternalProject)

set(TABULATE_PROJECT "tabulate")
set(TABULATE_REPO https://github.com/p-ranav/tabulate.git)
set(TABULATE_TAG b35db4cce50a4b296290b0ae827305cdeb23751e)
set(TABULATE_DIR ${THIRD_PARTY_PATH}/tabulate)
ExternalProject_Add(
  ${TABULATE_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY ${TABULATE_REPO}
  GIT_TAG ${TABULATE_TAG}
  DEPENDS ""
  PREFIX ${TABULATE_DIR}
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND "")

set(TABULATE_PATH ${TABULATE_DIR}/src/tabulate/single_include)
include_directories(${TABULATE_PATH})
