# PaddlePaddle Custom Device Implementation for Biren GPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Biren GPU.

## Compile and Install

```bash
# Acquire Biren PaddlePaddle Docker Image

# Clone PaddleCustomDevice source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# Compile Source Code and Install
cd backends/biren_gpu
mkdir -p build
pushd build
cmake -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug ..
ninja
pip3 install --no-index --find-links=offline dist/paddle_custom_supa-*.whl --force-reinstall
```

## Verification

```bash
# build with -DWITH_TESTING=ON
cmake -G Ninja -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug ..

# ctest
cd build
ninja test
```
