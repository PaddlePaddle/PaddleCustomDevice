#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# Usage:
# export CANN_VERSION=6.0.1
# bash build-image.sh ${CANN_VERSION}

CANN_VERSION=${1:-6.0.1} # default 6.0.1
CANN_TOOLKIT=Ascend-cann-toolkit_${CANN_VERSION}_linux-$(uname -m).run

DOCKER_VERSION=${CANN_VERSION//[^0-9]/} # 601
#DOCKER_VERSION=${CANN_VERSION//[^0-9a-z]/} # 60RC1
#DOCKER_VERSION=${DOCKER_VERSION,,} # lower case

# download $(uname -m) pkgs
if [ ! -f ${CANN_TOOLKIT} ]; then
  wget -q https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/${CANN_VERSION}/${CANN_TOOLKIT}
fi

# copy file to current directory
if [ ! -f ascend_install.info ]; then
  cp /etc/ascend_install.info ./
fi
if [ ! -f version.info ]; then
  cp /usr/local/Ascend/driver/version.info ./
fi

# ubuntu18-$(uname -m)-gcc82
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-$(uname -m)-gcc82
docker build --network=host -f Dockerfile.npu.ubuntu18.$(uname -m).gcc82 \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
     -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu18-$(uname -m)-gcc82 .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu18-$(uname -m)-gcc82

# kylinv10-$(uname -m)-gcc82
docker pull registry.baidubce.com/device/paddle-cpu:kylinv10-$(uname -m)-gcc82
docker build --network=host -f Dockerfile.npu.kylinv10.$(uname -m).gcc82 \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
    -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-kylinv10-$(uname -m)-gcc82 .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-kylinv10-$(uname -m)-gcc82

if [ $(uname -i) == 'aarch64' ]; then
# euleros-$(uname -m)-gcc82
  docker pull registry.baidubce.com/device/paddle-cpu:euleros-$(uname -m)-gcc82
  docker build --network=host -f Dockerfile.npu.euleros.$(uname -m).gcc82 \
    --build-arg CANN_VERSION=${CANN_VERSION} \
    --build-arg http_proxy=${proxy} \
    --build-arg https_proxy=${proxy} \
    --build-arg ftp_proxy=${proxy} \
    --build-arg no_proxy=bcebos.com \
    -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-euleros-$(uname -m)-gcc82 .
  docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-euleros-$(uname -m)-gcc82
fi
