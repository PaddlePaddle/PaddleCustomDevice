#!/bin/bash

# ubuntu18-aarch64-gcc82 
docker build --network=host -f Dockerfile.ubuntu18-aarch64-gcc82 \
      --build-arg http_proxy=${proxy} \
      --build-arg https_proxy=${proxy} \
      --build-arg ftp_proxy=${proxy} \
      --build-arg no_proxy=bcebos.com \
      -t registry.baidubce.com/device/paddle-cpu:ubuntu18-aarch64-gcc82 .
docker push registry.baidubce.com/device/paddle-cpu:ubuntu18-aarch64-gcc82

