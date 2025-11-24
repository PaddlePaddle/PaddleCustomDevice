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

set -e

# install requirement.txt
pip install -r requirement.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# uninstall paddle
pip  uninstall paddlepaddle -y


# init paddle
# git submodule sync --recursive && git submodule update --init --recursive


# pip install parameterized safetensors==0.6.2 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package
# install paddle


python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/


# apply patch
bash change_patch.sh


export MACA_PATH=/opt/maca
export CUDA_PATH=/workspace/cuda-11.7/
export PATH=${CUDA_PATH}/bin:${PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${PATH}:${CUCC_PATH}/tools:${CUCC_PATH}/bin
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

if [ ! -d build ]; then
    echo "build directory not found, creating..."
    mkdir build
fi

echo "make_maca"
cd build
cmake_maca .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) -DWITH_GPU=ON
make_maca -j60

echo "install whl"
pip install dist/paddle_metax_gpu*.whl --force-reinstall
cd ..
echo "Done!"
