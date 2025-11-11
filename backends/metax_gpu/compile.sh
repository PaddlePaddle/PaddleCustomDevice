# 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
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

export MACA_PATH=/opt/maca
export CUDA_PATH=/workspace/cuda-11.7/
export PATH=${CUDA_PATH}/bin:${PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${PATH}:${CUCC_PATH}/tools:${CUCC_PATH}/bin
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PADDLE_VERSION="3.3.0.dev$(date +%Y%m%d)"
if [ ! -d build ]; then
    echo "build directory not found, creating..."
    mkdir build
fi

echo "make_maca"
cd build
cmake_maca .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE=$(which python3) -DWITH_GPU=ON
make_maca -j18


echo "install whl"
pip install dist/paddle_metax_gpu*.whl --force-reinstall
cd ..
echo "Done!"
