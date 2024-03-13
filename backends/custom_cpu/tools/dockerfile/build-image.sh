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

# ubuntu20.gcc84-py38
docker build --network=host -f Dockerfile.ubuntu20.$(uname -m).gcc84 \
  --build-arg PY_VERSION=3.8 \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com,baidu-int.com \
  -t iregistry.baidu-int.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py38 .
docker push iregistry.baidu-int.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py38

# ubuntu20.gcc84-py310
docker build --network=host -f Dockerfile.ubuntu20.$(uname -m).gcc84 \
  --build-arg PY_VERSION=3.10 \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com,baidu-int.com \
  -t registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py310 .
docker push registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py310

# ubuntu18.gcc82-py310
docker build --network=host -f Dockerfile.ubuntu18.$(uname -m).gcc82 \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py310 .
docker push registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py310

# kylinv10.gcc82-py310
docker build --network=host -f Dockerfile.kylinv10.$(uname -m).gcc82 \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t registry.baidubce.com/device/paddle-cpu:kylinv10-$(uname -m)-gcc82-py310 .
docker push registry.baidubce.com/device/paddle-cpu:kylinv10-$(uname -m)-gcc82-py310

if [ $(uname -i) == 'x86_64' ]; then
  # kylinv10.gcc73-py310
  docker build --network=host -f Dockerfile.kylinv10.$(uname -m).gcc73 \
    --build-arg http_proxy=${proxy} \
    --build-arg https_proxy=${proxy} \
    --build-arg ftp_proxy=${proxy} \
    --build-arg no_proxy=bcebos.com \
    -t registry.baidubce.com/device/paddle-cpu:kylinv10-$(uname -m)-gcc73-py310 .
  docker push registry.baidubce.com/device/paddle-cpu:kylinv10-$(uname -m)-gcc73-py310
else
  # euleros.gcc82-py310
  docker build --network=host -f Dockerfile.euleros.$(uname -m).gcc82 \
    --build-arg http_proxy=${proxy} \
    --build-arg https_proxy=${proxy} \
    --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
    -t registry.baidubce.com/device/paddle-cpu:euleros-$(uname -m)-gcc82-py310 .
  docker push registry.baidubce.com/device/paddle-cpu:euleros-$(uname -m)-gcc82-py310
fi
