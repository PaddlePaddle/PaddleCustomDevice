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


#!/bin/bash

# update_paddle_dev.sh

chown -R $USER:$USER ../../Paddle/
chown -R $USER:$USER ../../../PaddleCustomDevice/
# Step 1: 撤销所有本地修改（已跟踪的文件，不包括新文件）
cd ../../Paddle/
echo "🔄 正在撤销所有本地修改（git checkout .）..."
git checkout develop
git checkout .

# Step 2: 拉取远程最新的 dev (通常是 develop) 分支代码
echo "🌐 正在拉取远程最新的 dev (develop) 分支代码..."


# 拉取 develop 分支的最新代码（与远程同步）
git pull origin develop

echo "🔗 当前分支: $(git branch --show-current)"
echo "📌 最新 commit hash (短): $(git rev-parse --short HEAD)"
echo "📌 最新 commit 信息:"
git log -1 --oneline

# 提示完成
echo "✅ 脚本执行完毕！"
echo "📌 已撤销本地修改，并更新到 Paddle 最新的 develop (dev) 分支代码。"


# pip install parameterized safetensors==0.6.2 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package
# install paddle

python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/


# unset http_proxy https_proxy
cd -
# apply patch
bash change_patch.sh

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
cmake_maca .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) -DWITH_GPU=ON
make_maca -j60

echo "install whl"
pip install dist/paddle_metax_gpu*.whl --force-reinstall
cd ..
echo "Done!"

# cd build/dist/
# ossutil ls oss://opensource-ci/paddle/
# ossutil cat oss://opensource-ci/paddle/

# ossutil cp ./paddle_metax_gpu-*.whl oss://opensource-ci/paddle/ -f
# cd -
