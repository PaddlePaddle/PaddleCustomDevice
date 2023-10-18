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

set -ex

# Usage:
# export ${FTP_USER} ${FTP_PASSWORD} ${CNTOOLKIT_VERSION} ${CNNL_VERSION} ${CNNL_EXTRA_VERSION} ${CNCL_VERSION} ${MLUOPS_VERSION}
# bash build-image.sh ${FTP_USER} ${FTP_PASSWORD} ${CNTOOLKIT_VERSION} ${CNNL_VERSION} ${CNNL_EXTRA_VERSION} ${CNCL_VERSION} ${MLUOPS_VERSION}

FTP_USER=${1} # Please contact Cambricon technicians to obtain username and password
FTP_PASSWORD=${2}
CNTOOLKIT_VERSION=${3:-3.6.1-1} # default 3.6.1
CNNL_VERSION=${4:-1.20.4-1} # default 1.20.4
CNCL_VERSION=${5:-1.11.0-1} # default 1.11.0
MLUOPS_VERSION=${6:-0.8.1-1} # default 0.8.1
CNNL_EXTRA_VERSION=${7:-1.4.1-1} # default 1.4.1-1

if [ $(uname -i) == 'x86_64' ]; then
  # ubuntu18-$(uname -m)-gcc82
  docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-$(uname -m)-gcc82
  docker build --network=host -f Dockerfile.mlu.ubuntu18.$(uname -m).gcc82 \
       --build-arg CNTOOLKIT_VERSION=${CNTOOLKIT_VERSION} \
       --build-arg CNNL_VERSION=${CNNL_VERSION} \
       --build-arg CNNL_EXTRA_VERSION=${CNNL_EXTRA_VERSION} \
       --build-arg CNCL_VERSION=${CNCL_VERSION} \
       --build-arg MLUOPS_VERSION=${MLUOPS_VERSION} \
       --build-arg FTP_USER=${FTP_USER} \
       --build-arg FTP_PASSWORD=${FTP_PASSWORD} \
       --build-arg http_proxy=${proxy} \
       --build-arg https_proxy=${proxy} \
       --build-arg ftp_proxy=${proxy} \
       --build-arg no_proxy=bcebos.com \
       -t registry.baidubce.com/device/paddle-mlu:cntoolkit${CNTOOLKIT_VERSION}-cnnl${CNNL_VERSION}-gcc82 .
  docker push registry.baidubce.com/device/paddle-mlu:cntoolkit${CNTOOLKIT_VERSION}-cnnl${CNNL_VERSION}-gcc82
else
  # kylinv10-$(uname -m)-gcc82
  docker pull registry.baidubce.com/device/paddle-cpu:kylinv10-$(uname -m)-gcc82
  docker build --network=host -f Dockerfile.mlu.kylinv10.$(uname -m).gcc82 \
       --build-arg FTP_USER=${FTP_USER} \
       --build-arg FTP_PASSWORD=${FTP_PASSWORD} \
       --build-arg http_proxy=${proxy} \
       --build-arg https_proxy=${proxy} \
       --build-arg ftp_proxy=${proxy} \
       --build-arg no_proxy=bcebos.com \
       -t registry.baidubce.com/device/paddle-mlu:kylinv10-$(uname -m)-gcc82 .
  docker push registry.baidubce.com/device/paddle-mlu:kylinv10-$(uname -m)-gcc82
fi
