# PaddlePaddle Custom Device Implementation for Iluvatar GPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Iluvatar GPU.

## Compile and Install

```bash
# Acquire Iluvatar PaddlePaddle Docker Image

# Clone PaddleCustomDevice source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# Compile Source Code
cd backends/iluvatar_gpu
bash build_paddle.sh

# Install PaddlePaddle
bash install_paddle.sh
```

## Verification

```bash
# build with BUILD_TEST=1

# run_test
cd tests
bash run_test.sh
```
