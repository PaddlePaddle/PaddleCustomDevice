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

include(ExternalProject)

set(GIT_URL "https://github.com")
set(THIRD_PARTY_PATH  "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
    "A path setting third party libraries download & build directories.")

set(THIRD_PARTY_BUILD_TYPE Release)
set(third_party_deps)

########################### include third_party ###############################
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/pybind11)

# graph executor dependencies
macro(UNSET_VAR VAR_NAME)
  unset(${VAR_NAME} CACHE)
  unset(${VAR_NAME})
endmacro()
include(external/zlib)
include(external/xxhash)
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
set(TEMP ${CMAKE_CXX_FLAGS})
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=1" "-D_GLIBCXX_USE_CXX11_ABI=0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0)
include(external/protobuf)
set(CMAKE_CXX_FLAGS ${TEMP})

list(APPEND third_party_deps extern_gflags extern_glog extern_pybind)

if (WITH_TESTING)
    include(external/gtest)     # download, build, install gtest
    list(APPEND third_party_deps extern_gtest)
endif()

if (WITH_MKLDNN AND NOT WITH_ARM)
    include(external/mkldnn)    # download, build, install mkldnn
    list(APPEND third_party_deps extern_mkldnn)
endif()

add_custom_target(third_party ALL DEPENDS ${third_party_deps})
