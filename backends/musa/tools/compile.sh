#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

SOURCE_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

# prepare build directory
mkdir -p ${SOURCE_ROOT}/build
cd ${SOURCE_ROOT}/build

arch=$(uname -i)
if [ $arch == 'x86_64' ]; then
    WITH_MKL=ON
else
    WITH_MKL=OFF
fi

cat <<EOF
========================================
Configuring cmake in build ...
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
    -DWITH_MUSA=ON 
    -DWITH_MCCL=ON 
    -DWITH_TESTING=${WITH_TESTING:-ON}
    -DWITH_MKL=${WITH_MKL:-ON}
    -DON_INFER=${ON_INFER:-OFF}
    -DWITH_COVERAGE=${WITH_COVERAGE:-OFF}
    -DPYTHON_VERSION=${PYTHON_VERSION:-""}
========================================
EOF

set +e
cmake .. \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
    -DWITH_MUSA=ON \
    -DWITH_MCCL=ON \
    -DWITH_TESTING=${WITH_TESTING:-OFF} \
    -DWITH_MKL=${WITH_MKL:-ON} \
    -DON_INFER=${ON_INFER:-OFF} \
    -DWITH_CINN=OFF \
    -DWITH_COVERAGE=${WITH_COVERAGE:-OFF} \
    -DPY_VERSION=3.10 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON;cmake_error=$?

if [ "$cmake_error" != 0 ];then
    echo "CMake Error Found !!!"
    exit 7;
fi

if [ $arch == 'x86_64' ]; then
    make -j128;make_error=$?
else
    make TARGET=ARMV8 -j30;make_error=$?
fi

if [ "$make_error" != 0 ];then
    echo "Make Error Found !!!"
    exit 7;
fi
