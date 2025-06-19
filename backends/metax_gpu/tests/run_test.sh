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

# export PATH=/usr/local/corex-4.3.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/corex-4.3.0/lib
# export LIBRARY_PATH=/usr/local/corex-4.3.0/lib
# export PYTHONPATH=${PYTHONPATH}:${PADDLE_SOURCE_DIR}/test/legacy_test
pip install scipy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package
# export PYTHONPATH=../../../../PaddleCustomDevice/Paddle/test/legacy_test/:/../../../../PaddleCustomDevice/Paddle/test/:../../../../PaddleCustomDevice/python:../../../../PaddleCustomDevice/python/tests/
# export PYTHONPATH="../../../../PaddleCustomDevice/Paddle/test/legacy_test/:/../../../../PaddleCustomDevice/Paddle/test/:/m01097/PaddleCustomDevice/python:/m01097/PaddleCustomDevice/python/tests/"
export PYTHONPATH="../../../../PaddleCustomDevice/Paddle/test/legacy_test/:/../../../../PaddleCustomDevice/Paddle/test/:/m01097/PaddleCustomDevice/python:/m01097/PaddleCustomDevice/python/tests/"
# mkdir -p build && cd build && cmake ..
# make run_tests
# 进入 build 目录（如果不存在则创建）
mkdir -p build && cd build

# 生成构建系统（指定上级目录的 CMakeLists.txt）
cmake ..

# 编译（虽然 Python 测试无需编译，但 CTest 需要此步骤）
cmake --build .

# 运行所有测试
ctest --output-on-failure

# 可选：并行运行测试（N 是线程数）
# ctest -j4 --output-on-failure
