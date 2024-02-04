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
# export CANN_VERSION=7.0.0
# bash build-image.sh ${CANN_VERSION}

CANN_VERSION=${1:-7.0.0} # default 7.0.0

# DOCKER_VERSION=${CANN_VERSION//[^0-9]/} # 700
DOCKER_VERSION=${CANN_VERSION//[^0-9a-z]/} # 700
#DOCKER_VERSION=${DOCKER_VERSION,,} # lower case

# Download packages from https://www.hiascend.com/software/cann/community first
if [ ! -f Ascend-cann-toolkit_${CANN_VERSION}_linux-$(uname -m).run ]; then
  echo "Please download CANN installation packages first!"
  exit 1
fi

# copy file to current directory
cp /etc/ascend_install.info ./
cp /usr/local/Ascend/driver/version.info ./

# get chip version and fix HCCL_BUFFSIZE
CHIP_VERSION="UNKNOWN"
if [ $(lspci | grep d801 | wc -l) -ne 0 ]; then
  CHIP_VERSION="910A"
  sed -i "s/HCCL_BUFFSIZE=.*/HCCL_BUFFSIZE=60/g" Dockerfile.npu.*
elif [ $(lspci | grep d802 | wc -l) -ne 0 ]; then
  CHIP_VERSION="910B"
  sed -i "s/HCCL_BUFFSIZE=.*/HCCL_BUFFSIZE=120/g" Dockerfile.npu.*
else
  echo "Please make sure Ascend 910A or 910B NPUs exists!"
  exit 1
fi

# ubuntu18-$(uname -m)-gcc82
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-$(uname -m)-gcc82
docker build --network=host -f Dockerfile.npu.ubuntu18.$(uname -m).gcc82 \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-${CHIP_VERSION}-ubuntu18-$(uname -m) .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-${CHIP_VERSION}-ubuntu18-$(uname -m)

# kylinv10-$(uname -m)-gcc82
docker pull registry.baidubce.com/device/paddle-cpu:kylinv10-$(uname -m)-gcc82
docker build --network=host -f Dockerfile.npu.kylinv10.$(uname -m).gcc82 \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-${CHIP_VERSION}-kylinv10-$(uname -m) .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-${CHIP_VERSION}-kylinv10-$(uname -m)

if [ $(uname -i) == 'aarch64' ]; then
# euleros-$(uname -m)-gcc82
  docker pull registry.baidubce.com/device/paddle-cpu:euleros-$(uname -m)-gcc82
  docker build --network=host -f Dockerfile.npu.euleros.$(uname -m).gcc82 \
    --build-arg CANN_VERSION=${CANN_VERSION} \
    --build-arg http_proxy=${proxy} \
    --build-arg https_proxy=${proxy} \
    --build-arg ftp_proxy=${proxy} \
    --build-arg no_proxy=bcebos.com \
    -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-${CHIP_VERSION}-euleros-$(uname -m) .
  docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-${CHIP_VERSION}-euleros-$(uname -m)
fi

# clean driver info
rm -rf *.info
