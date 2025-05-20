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

set -e

echo "uninstall paddle-metax-gpu..."
pip uninstall -y paddle-metax-gpu

if [ ! -d build ]; then
    echo "build directory not found, creating..."
    mkdir build
fi

echo "make"
cd build
cmake .. -DPython3_EXECUTABLE=$(which python3) -DWITH_GPU=ON
make -j8

echo "install whl"
pip install dist/paddle_metax_gpu*.whl
cd ..
echo "Done!"
