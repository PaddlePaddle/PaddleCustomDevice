#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

PYTHON_VERSION=${PYTHON_VERSION:-$(python3 -V 2>&1|awk '{print $2}')}
COREX_VERSION=${COREX_VERSION:-latest}
if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
BUILD_TEST=${BUILD_TEST:-1}
COREX_ARCH=${COREX_ARCH:-ivcore11}
export CMAKE_CUDA_ARCHITECTURES=${COREX_ARCH}

CURRENT_DIR=$(pwd)
PADDLE_SOURCE_DIR="${CURRENT_DIR}/../../Paddle"
PATCH_FILE="${CURRENT_DIR}/patches/paddle-corex.patch"
# set BUILD_WITH_FLAGCX to 1 if we want to use flagcx as communication backend
BUILD_WITH_FLAGCX=0
FLAGCX_ROOT="/workspace/FlagCX"

if [ "$BUILD_WITH_FLAGCX" == "1" ]; then
    WITH_FLAGCX="ON"
else
    WITH_FLAGCX="OFF"
fi

if ! git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
  if ! git -C "$PADDLE_SOURCE_DIR" apply "$PATCH_FILE"; then
    echo "Error: Failed to apply patch!"
    exit 1
  fi
  echo "Patch applied successfully!"
fi
cp -r ${CURRENT_DIR}/patches/eigen/Core ../../Paddle/third_party/eigen3/Eigen/Core
cp -r ${CURRENT_DIR}/patches/eigen/Tensor ../../Paddle/third_party/eigen3/unsupported/Eigen/CXX11/Tensor
cp -r ${CURRENT_DIR}/patches/eigen/TensorAssign.h ../../Paddle/third_party/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorAssign.h

pushd ${PADDLE_SOURCE_DIR}/paddle/phi/core
if [[ ! -f "external_error.pb.cc" || ! -f "external_error.pb.h" ]]; then
  protoc --cpp_out=. external_error.proto
fi
popd

if [[ ! -d "build" ]]; then
  mkdir build
fi
pushd build

cmake -G Ninja -DPY_VERSION=${PYTHON_VERSION} -DWITH_COREX=ON -DPADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR} \
-DWITH_DISTRIBUTE=ON -DWITH_NCCL=ON -DWITH_FLAGCX=${WITH_FLAGCX} -DWITH_RCCL=OFF -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DON_INFER=ON -DCOREX_VERSION=${COREX_VERSION} -DCOREX_ARCH=${COREX_ARCH} \
-DFLAGCX_ROOT=${FLAGCX_ROOT} \
-DCMAKE_CXX_FLAGS='-Wno-error=pessimizing-move -Wno-error=deprecated-copy -Wno-error=init-list-lifetime -pthread' \
-DCMAKE_CUDA_FLAGS='-Xclang -fcuda-allow-variadic-functions -mllvm --skip-double' \
-DCMAKE_C_FLAGS="-pthread" \
-DWITH_ARM=OFF -DWITH_DGC=OFF .. 2>&1 | tee compile.log
ninja -k 0 -j$(nproc) 2>&1 | tee -a compile.log
FAILED_LOG="failed_files.log"
grep -E "FAILED: " compile.log | tee ${FAILED_LOG}
echo "Failed files are listed in ${FAILED_LOG}"
popd

if [[ ! -d "build_pip" ]]; then
  mkdir build_pip
fi
cp build/dist/* build_pip

if git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
  git -C "$PADDLE_SOURCE_DIR" apply --reverse "$PATCH_FILE"
  echo "Patch successfully reverted!"
fi

pushd ${PADDLE_SOURCE_DIR}/third_party/eigen3
git reset --hard
popd

exit 0
