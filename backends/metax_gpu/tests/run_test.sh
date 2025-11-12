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

pip install scipy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package
SCRIPT_DIR=$(dirname "$0")
LEGACY_TEST_PATH="${SCRIPT_DIR}/../../../Paddle/test/legacy_test"
TEST_PATH1="${SCRIPT_DIR}/../../../python"
TEST_PATH2="${SCRIPT_DIR}/../../../python/tests"
export PYTHONPATH="${LEGACY_TEST_PATH}:${PYTHONPATH}:${TEST_PATH1}:${TEST_PATH2}"
export PADDLE_XCCL_BACKEND=metax_gpu
export CUDA_VISIBLE_DEVICES=0
# export
# sleep 1000000


rm -r build
mkdir -p build && cd build


TEST_LOG_LEVEL=0
TEST_LIST_FILE=""
TEST_LOG_OUTPUT_DIR=""
TEST_PARALLEL_NUM=1

while getopts "i:o:v:j:h" opt; do
  case "$opt" in
    i)
      TEST_LIST_FILE="$OPTARG"
      ;;
    o)
      TEST_LOG_OUTPUT_DIR="$OPTARG"
      echo "Set log output dir [ $TEST_LOG_OUTPUT_DIR ]"
      ;;
    v)
      TEST_LOG_LEVEL=$OPTARG
      ;;
    j)
      TEST_PARALLEL_NUM="$OPTARG"
      ;;
    h)
      echo "用法：$0 -i <测试列表文件> -o <日志输出路径> ..."
      echo "选项说明："
      echo "  -i  测试程序列表文件"
      echo "  -o  日志输出路径"
      echo "  -v  GLOG_v 日志等级"
      echo "  -j  ctest 测试并行数量"
      echo "  -h  显示帮助"
      exit 0
      ;;
    \?)
      echo "error: unknow option '-$OPTARG'."
      exit 1
      ;;
    :)
      echo "error option '-$OPTARG' must have parameter."
      exit 1
      ;;
  esac
done


export GLOG_v=$TEST_LOG_LEVEL


cmake .. -DTEST_LIST_FILE=$TEST_LIST_FILE -DLOG_OUTPUT_DIR=$TEST_LOG_OUTPUT_DIR

cmake --build .

ctest -j$TEST_PARALLEL_NUM --output-on-failure
