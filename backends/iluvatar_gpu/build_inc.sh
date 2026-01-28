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
BUILD_DIR="${CURRENT_DIR}/build"
BUILD_PIP_DIR="${CURRENT_DIR}/build_pip"
STATE_FILE="${CURRENT_DIR}/.build_state"
# set BUILD_WITH_FLAGCX to 1 if we want to use flagcx as communication backend
BUILD_WITH_FLAGCX=0
FLAGCX_ROOT="/workspace/FlagCX"
PLATFORM_ID=$(uname -i)

if [ "$BUILD_WITH_FLAGCX" == "1" ]; then
    WITH_FLAGCX="ON"
else
    WITH_FLAGCX="OFF"
fi

# Check if cleanup is needed
if [[ "$1" == "--clean" ]]; then
    echo "Cleaning build environment..."
    
    # Clean build directories
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
        echo "Removed build directory"
    fi
    
    if [[ -d "$BUILD_PIP_DIR" ]]; then
        rm -rf "$BUILD_PIP_DIR"
        echo "Removed build_pip directory"
    fi
    
    # Revert patch
    if git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
        git -C "$PADDLE_SOURCE_DIR" apply --reverse "$PATCH_FILE"
        echo "Patch successfully reverted!"
    else
        echo "Patch was not applied or already reverted"
    fi
    
    # Reset eigen
    pushd ${PADDLE_SOURCE_DIR}/third_party/eigen3 > /dev/null
    git reset --hard
    popd > /dev/null
    echo "Eigen repository reset"
    
    # Delete state file
    if [[ -f "$STATE_FILE" ]]; then
        rm -f "$STATE_FILE"
        echo "Removed state file"
    fi
    
    echo "Clean completed!"
    exit 0
fi

# Check if this is a first-time build
if [[ ! -f "$STATE_FILE" ]]; then
    echo "First time build detected. Setting up environment..."
    
    # Update git submodules
    echo "Updating git submodules..."
    pushd ${CURRENT_DIR}/../../ > /dev/null
    git submodule update --init --recursive --force
    popd > /dev/null
    echo "Git submodules updated"
    
    # Apply patch
    if ! git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
        if ! git -C "$PADDLE_SOURCE_DIR" apply "$PATCH_FILE"; then
            echo "Error: Failed to apply patch!"
            exit 1
        fi
        echo "Patch applied successfully!"
    else
        echo "Patch was already applied"
    fi
    
    # Copy eigen files
    cp -r ${CURRENT_DIR}/patches/eigen/Core ../../Paddle/third_party/eigen3/Eigen/Core
    cp -r ${CURRENT_DIR}/patches/eigen/Tensor ../../Paddle/third_party/eigen3/unsupported/Eigen/CXX11/Tensor
    cp -r ${CURRENT_DIR}/patches/eigen/TensorAssign.h ../../Paddle/third_party/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorAssign.h
    echo "Eigen files copied successfully"
    
    # Generate protobuf files
    pushd ${PADDLE_SOURCE_DIR}/paddle/phi/core > /dev/null
    if [[ ! -f "external_error.pb.cc" || ! -f "external_error.pb.h" ]]; then
        protoc --cpp_out=. external_error.proto
        echo "Protobuf files generated"
    else
        echo "Protobuf files already exist"
    fi
    popd > /dev/null
    
    # Create state file to mark environment as set up
    echo "BUILD_ENV_SET=1" > "$STATE_FILE"
    echo "Environment setup completed"
else
    echo "Incremental build detected. Skipping environment setup."
fi

# Create build directory
if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir -p "$BUILD_DIR"
    echo "Created build directory"
fi

# Enter build directory
pushd "$BUILD_DIR" > /dev/null

# If this is a first-time build, run cmake
rm -f compile.log
if [[ ! -f "Makefile" && ! -f "build.ninja" ]]; then
    echo "Running cmake for first time build..."
    if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
        cmake -G Ninja -DPY_VERSION=${PYTHON_VERSION} -DWITH_COREX=ON -DPADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR} \
        -DWITH_DISTRIBUTE=ON -DWITH_NCCL=ON -DWITH_FLAGCX=${WITH_FLAGCX} -DWITH_RCCL=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DON_INFER=ON -DCOREX_VERSION=${COREX_VERSION} -DCOREX_ARCH=${COREX_ARCH} \
        -DFLAGCX_ROOT=${FLAGCX_ROOT} \
        -DCMAKE_CXX_FLAGS='-Wno-error=pessimizing-move -Wno-error=deprecated-copy -Wno-error=init-list-lifetime -pthread' \
        -DCMAKE_CUDA_FLAGS='-Xclang -fcuda-allow-variadic-functions -mllvm --skip-double' \
        -DCMAKE_C_FLAGS="-pthread" \
        -DWITH_ARM=ON -DWITH_DGC=OFF .. 2>&1 | tee compile.log
    else
        cmake -G Ninja -DPY_VERSION=${PYTHON_VERSION} -DWITH_COREX=ON -DPADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR} \
        -DWITH_DISTRIBUTE=ON -DWITH_NCCL=ON -DWITH_FLAGCX=${WITH_FLAGCX} -DWITH_RCCL=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DON_INFER=ON -DCOREX_VERSION=${COREX_VERSION} -DCOREX_ARCH=${COREX_ARCH} \
        -DFLAGCX_ROOT=${FLAGCX_ROOT} \
        -DCMAKE_CXX_FLAGS='-Wno-error=pessimizing-move -Wno-error=deprecated-copy -Wno-error=init-list-lifetime -pthread' \
        -DCMAKE_CUDA_FLAGS='-Xclang -fcuda-allow-variadic-functions -mllvm --skip-double' \
        -DCMAKE_C_FLAGS="-pthread" \
        -DWITH_ARM=OFF -DWITH_DGC=OFF .. 2>&1 | tee compile.log
    fi
else
    echo "Cmake configuration already exists, skipping..."
fi

# Compile
echo "Starting compilation..."
if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
    env TARGET=ARMV8 ninja -j$(nproc) 2>&1
else
    ninja -j$(nproc) 2>&1
fi
FAILED_LOG="failed_files.log"
grep -E "FAILED: " compile.log | tee ${FAILED_LOG}
echo "Failed files are listed in ${FAILED_LOG}"

# Return to parent directory
popd > /dev/null

# Create build_pip directory
if [[ ! -d "$BUILD_PIP_DIR" ]]; then
    mkdir -p "$BUILD_PIP_DIR"
fi

# Copy build artifacts
cp build/dist/* build_pip
echo "Build artifacts copied to build_pip directory"

# Uninstall old version of paddle-iluvatar-gpu
echo "Uninstalling old paddle-iluvatar-gpu..."
pip uninstall -y paddle-iluvatar-gpu 2>/dev/null || echo "paddle-iluvatar-gpu not installed or already uninstalled"

# Install newly compiled paddle-iluvatar-gpu
echo "Installing new paddle-iluvatar-gpu..."
pip install build_pip/*.whl --force-reinstall

echo "Build and installation completed successfully!"
exit 0
