# PaddlePaddle Custom Device Implementation for Iluvatar GPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Iluvatar GPU.

## Compilation and Installation

```bash
# Please contact Iluvatar customer support (services@iluvatar.com) to obtain the SDK image

# Clone PaddleCustomDevice source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# Set environment variables
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib
export LIBRARY_PATH=/usr/local/corex/lib

# Compile Paddle Custom Device
cd backends/iluvatar_gpu
bash build_paddle.sh

# Install
bash install_paddle.sh
```
## For incremental compilation（faster rebuilds after code changes）
```bash
# For incremental compilation (faster rebuilds after code changes, also installs whl)
bash build_inc.sh

# Clean build environment (removes build directories, reverts patches, and resets state)
bash build_inc.sh --clean
```

## Verification

```bash
# Run tests
cd tests
bash run_test.sh
```
