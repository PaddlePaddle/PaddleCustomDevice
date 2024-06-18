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

#!/bin/bash
set +x
set -e

# 1. checkout Paddle to commit b065877d
old_path=${PWD}
cd $old_path/../../Paddle/
git checkout b065877d
cd $old_path

# 2. Prepare build directory
build_dir=$old_path/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

# 3. Configure options
export WITH_TESTING=ON
export SDAA_ROOT=/opt/tecoai # no default path
export SDPTI_ROOT=/opt/tecoai # no default path
export TECODNN_ROOT=/opt/tecoai # no default path
export EXTEND_OP_ROOT=/opt/tecoai/extend # no default path
export TBLAS_ROOT=/opt/tecoai # no default path
export TCCL_ROOT=/opt/tecoai # no default path
export TECODNN_CUSTOM_ROOT=/opt/tecoai # no default path
export PADDLE_SOURCE_DIR=$old_path/../../Paddle # default is "../../Paddle", required if WITH_TESTING is ON

# 4. CMake command
cmake .. -DNATIVE_SDAA=ON \
  -DPython_EXECUTABLE=`which python3` \
  -DWITH_TESTING=${WITH_TESTING} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_VERBOSE_MAKEFILE=ON

# 5. Make command
make -j
