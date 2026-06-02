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

rm -r ../../Paddle/third_party/eigen3
cd patch
unzip Eigen_3.4.0_paddle.zip
mv Eigen_3.4.0_paddle eigen3
cd ..
cp -r patch/eigen3/ ../../Paddle/third_party/eigen3
rm -r patch/eigen3
# cp patch/tmp/mixed_vector* ../../Paddle/paddle/phi/core
cd ../../Paddle/
git apply --verbose ../backends/metax_gpu/patch/paddle.patch
cd -
# cp -r patch/intrinsics.cuh ../../Paddle/third_party/warpctc/include/contrib/moderngpu/include/device/
cp -r ./patch/warpctc.patch ../../Paddle/third_party/warpctc/
