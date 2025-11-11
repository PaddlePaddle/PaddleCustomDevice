# PaddlePaddle Custom Device Implementaion for Custom Intel GPU

Please refer to the following steps to compile, install and verify the custom device implementaion for Custom Intel GPU.

## Activate oneapi env vars

```bash
source load.sh
```

## Get Sources

```bash
# clone source
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# get the latest submodule source code
git submodule sync
git submodule update --remote --init --recursive
```

## Compile and Install

```bash
# navigate to implementaion for Custom CPU
cd backends/intel_gpu

# before compiling, ensure that Paddle is installed, you can run the following command
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# create the build directory and navigate in
mkdir build && cd build

cmake ..
make -j $(nproc)

# using pip to install the output
pip install dist/paddle_custom_intel_gpu*.whl
```

## Verification

```bash
# check the plugin status
python -c "import paddle; print('intel_gpu' in paddle.device.get_all_custom_device_type())"

# expected output
True

```
