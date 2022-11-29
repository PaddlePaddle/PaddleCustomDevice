#!/bin/bash

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

set -x

# Usage:
# export CANN_VERSION=6.0.0.alpha002
# bash build-x86_64.sh ${CANN_VERSION}

CANN_VERSION=${1:-6.0.0.alpha002} # default 6.0.0.alpha002
CANN_TOOLKIT=Ascend-cann-toolkit_${CANN_VERSION}_linux-x86_64.run

DOCKER_VERSION=${CANN_VERSION//[^0-9]/}
DOCKER_VERSION=${DOCKER_VERSION:0:3}

# download x86_64 pkgs
if [ ! -f ${CANN_TOOLKIT} ]; then
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/${CANN_VERSION}/${CANN_TOOLKIT}
fi

# copy file to current directory
if [ ! -f ascend_install.info ]; then
    cp /etc/ascend_install.info ./
fi
if [ ! -f version.info ]; then
    cp /usr/local/Ascend/driver/version.info ./
fi

# ubuntu18-x86_64-gcc82
export proxy=http://172.19.57.45:3128
# export proxy=http://172.19.56.199:3128
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-x86_64-gcc82
docker build --network=host -f Dockerfile.npu.ubuntu18-x86_64-gcc82 \
       --build-arg CANN_VERSION=${CANN_VERSION} \
       --build-arg http_proxy=${proxy} \
       --build-arg https_proxy=${proxy} \
       --build-arg ftp_proxy=${proxy} \
       --build-arg no_proxy=bcebos.com \
       -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-x86_64-gcc82 .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-x86_64-gcc82
