#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# BOTH for x86_64 and aarch64

# ubuntu20.gcc84-py39
docker build --network=host -f Dockerfile.ubuntu20.$(uname -m).gcc84 \
  --build-arg PY_VERSION=3.9 \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com,baidu-int.com \
  -t registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py39 .
docker push registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py39
