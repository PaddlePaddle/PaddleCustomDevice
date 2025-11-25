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

SCRIPT_DIR=$(dirname "$(realpath "$0")")
LEGACY_TEST_PATH="${SCRIPT_DIR}/../../../Paddle/test/legacy_test"
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib
export LIBRARY_PATH=/usr/local/corex/lib
export PYTHONPATH="${LEGACY_TEST_PATH}:${PYTHONPATH}"

if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    echo "ERROR: LD_LIBRARY_PATH is not set!" >&2
    exit 1
elif [[ ! -f "${LD_LIBRARY_PATH}/libcuda.so.1" ]]; then
    echo "ERROR: libcuda.so.1 not found in LD_LIBRARY_PATH!" >&2
    exit 1
fi

NUM_GPUS=$(ixsmi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPU found!"
    exit 1
fi
LAST_GPU=$((NUM_GPUS - 1))
echo "Using last GPU: $LAST_GPU"
export CUDA_VISIBLE_DEVICES=$LAST_GPU

export LD_PRELOAD="${LD_LIBRARY_PATH}/libcuda.so.1"
export FLAG_SKIP_FLOAT64=1

CURRENT_DIR=$(pwd)
PADDLE_SOURCE_DIR="${CURRENT_DIR}/../../../Paddle"
PATCH_FILE="${CURRENT_DIR}/../patches/paddle-corex-test.patch"

if ! git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
  if ! git -C "$PADDLE_SOURCE_DIR" apply "$PATCH_FILE"; then
    echo "Error: Failed to apply patch!"
    exit 1
  fi
  echo "Patch applied successfully!"
fi

mkdir -p build || { echo "ERROR: Failed to create build directory"; exit 1; }
cd build || { echo "ERROR: Failed to enter build directory"; exit 1; }

echo "=== Configuring project ==="
cmake .. || { echo "ERROR: CMake configuration failed"; exit 1; }

echo "=== Building project (if needed) ==="
make -j$(nproc) || { echo "ERROR: Build failed"; exit 1; }

echo "=== Running tests ==="
ctest --output-on-failure -V -j8 || { 
    echo "ERROR: Tests failed!" >&2
    echo "Exit code: $?" >&2
    exit 1
}

echo "=== All tests passed successfully ==="

if git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
  git -C "$PADDLE_SOURCE_DIR" apply --reverse "$PATCH_FILE"
  echo "Patch successfully reverted!"
fi

cd - > /dev/null
