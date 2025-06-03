# PaddlePaddle Custom Device Implementation for METAX GPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Iluvatar GPU.

## Install Paddle

python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

## Compile and Install

```bash
# Acquire Metax PaddlePaddle Docker Image

# Clone PaddleCustomDevice source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# Compile Source Code
cd backends/metax_gpu
bash build.sh

# Install PaddlePaddle
# bash install_paddle.sh
```

## Verification

```bash
# build with BUILD_TEST=1

# run_test
cd tests
bash run_test.sh
```
