# PaddlePaddle Custom Device Implementation for Biren GPU

English | [简体中文](./README.md)

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
pip3 install --no-index --find-links=offline dist/paddle_custom_supa-1.0.0.0*.whl --force-reinstall
```

## Verification

```bash
python3 tests/test_abs_op_supa.py
```