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
# export CANN_VERSION=8.0.RC1
# bash build-image.sh ${CANN_VERSION}

CANN_VERSION=${1:-8.0.RC1} # default 8.0.RC1

# DOCKER_VERSION=${CANN_VERSION//[^0-9]/} # 801
DOCKER_VERSION=${CANN_VERSION//[^0-9A-Z]/} # 80RC1
#DOCKER_VERSION=${DOCKER_VERSION,,} # lower case

# Download packages from https://www.hiascend.com/software/cann/community first
if [ ! -f Ascend-cann-toolkit_${CANN_VERSION}_linux-$(uname -m).run ]; then
  echo "Please download CANN installation packages first!"
  exit 1
fi

# copy file to current directory
cp /etc/ascend_install.info ./
cp /usr/local/Ascend/driver/version.info ./

# ubuntu20-$(uname -m)-gcc84-py38
sed "s#<baseimg>#registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py38#g" Dockerfile.npu.ubuntu20.gcc84 > Dockerfile.npu.ubuntu20.gcc84.py38
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py38
docker build --network=host -f Dockerfile.npu.ubuntu20.gcc84.py38 \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-$(uname -m)-gcc84-py38 .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-$(uname -m)-gcc84-py38
rm -rf Dockerfile.npu.ubuntu20.gcc84.py38

# ubuntu20-$(uname -m)-gcc84-py39
sed "s#<baseimg>#registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py39#g" Dockerfile.npu.ubuntu20.gcc84 > Dockerfile.npu.ubuntu20.gcc84.py39
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py39
docker build --network=host -f Dockerfile.npu.ubuntu20.gcc84.py39 \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-$(uname -m)-gcc84-py39 .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-$(uname -m)-gcc84-py39
rm -rf Dockerfile.npu.ubuntu20.gcc84.py39

# ubuntu20-$(uname -m)-gcc84-py310
sed "s#<baseimg>#registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py310#g" Dockerfile.npu.ubuntu20.gcc84 > Dockerfile.npu.ubuntu20.gcc84.py310
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu20-$(uname -m)-gcc84-py310
docker build --network=host -f Dockerfile.npu.ubuntu20.gcc84.py310 \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-$(uname -m)-gcc84-py310 .
docker push registry.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-$(uname -m)-gcc84-py310
rm -rf Dockerfile.npu.ubuntu20.gcc84.py310


# clean driver info
rm -rf *.info
