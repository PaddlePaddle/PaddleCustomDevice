# PaddlePaddle Custom Device Implementation for Ascend NPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Ascend NPU.

## Prepare environment and source code

> Note: [CANN 6.0.1](https://www.hiascend.com/software/cann/community-history?id=6.0.1.alpha001) is supported.

```bash
# 1. pull PaddlePaddle Ascend NPU development docker image
# dockerfile of the image is in tools/dockerfile directory
docker pull registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-x86_64-gcc82
docker pull registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-aarch64-gcc82
docker pull registry.baidubce.com/device/paddle-npu:cann601-kylinv10-aarch64-gcc82

# 2. refer to the following commands to start docker container
docker run -it --name paddle-npu-dev -v `pwd`:/workspace \
       --workdir=/workspace --pids-limit 409600 \
       --privileged --network=host --shm-size=128G \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
       -v /usr/local/dcmi:/usr/local/dcmi \
       registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-$(uname -m)-gcc82 /bin/bash

# 3. clone the source code recursively along with Paddle source code
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 4. execute the following commands to update submodule
git submodule sync
git submodule update --remote --init --recursive
```

## PaddlePaddle Installation and Verification

### Source Code Compile

```bash
# 1. go to ascend npu directory
cd backends/npu

# 2. please ensure the PaddlePaddle cpu whl package is already installed
# the development docker image NOT have PaddlePaddle cpu whl installed by default
# you may download and install the nightly built cpu whl package with links below
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp37-cp37m-linux_aarch64.whl

# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script
bash tools/compile.sh

# 5. install the generated package, which is under build/dist directory
pip install build/dist/paddle_custom_npu*.whl
```

## Verification

```bash
# 1. list available hardware backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# expected output
['npu']

# 2. verify a simple model training
python tests/test_MNIST_model.py
# expected output
... ...
Epoch 0 step 0, Loss = [2.3313463], Accuracy = 0.046875
Epoch 0 step 100, Loss = [1.9624571], Accuracy = 0.484375
Epoch 0 step 200, Loss = [2.002725], Accuracy = 0.453125
Epoch 0 step 300, Loss = [1.912869], Accuracy = 0.546875
Epoch 0 step 400, Loss = [1.9169667], Accuracy = 0.5625
Epoch 0 step 500, Loss = [1.9007692], Accuracy = 0.5625
Epoch 0 step 600, Loss = [1.8512673], Accuracy = 0.625
Epoch 0 step 700, Loss = [1.8759218], Accuracy = 0.59375
Epoch 0 step 800, Loss = [1.8942316], Accuracy = 0.5625
Epoch 0 step 900, Loss = [1.8966292], Accuracy = 0.5625
```

## PaddleInference C++ Installation and Verification

### PaddleInference C++ Source Compile

> Note: the official released PaddleInference C++ package do not support custom device, please follow the steps below to source compile PaddleInference C++ package.

```bash
# 1. got ot Paddle source code directory
cd PaddleCustomDevice/Paddle

# 2. prepare build directory
mkdir build && cd build

# 3.1 build command for X86_64
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_ARM=OFF
make -j8

# 3.2 build command for aarch64
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_ARM=ON
make TARGET=ARMV8 -j8

# 4) PaddleInference C++ package will be generated into build/paddle_inference_install_dir directory
```

### Ascend NPU Inference Source Compile
```bash
# 1. go to ascend npu directory
cd backends/npu

# 2. compile options, the PADDLE_INFERENCE_LIB_DIR is the path of Paddle Inference C++ package
# generated in the previous step, i.e. build/paddle_inference_install_dir directory
export ON_INFER=ON # whether to enable C++ inference, default is OFF
export PADDLE_INFERENCE_LIB_DIR=/path/to/Paddle/build/paddle_inference_install_dir

# 3. execute compile script
bash tools/compile.sh

# 4. Specify CUSTOM_DEVICE_ROOT to the folder of libpaddle-custom-npu.so
export CUSTOM_DEVICE_ROOT=/path/to/PaddleCustomDevice/backends/npu/build
```

### Ascend NPU Inference Verification

```bash
# 1. clone Paddle-Inference-Demo source code
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git

# 2. Copy the PaddleInference C++ package to Paddle-Inference-Demo/c++/lib
cp -r PaddleCustomDevice/Paddle/build/paddle_inference_install_dir Paddle-Inference-Demo/c++/lib/paddle_inference
# directory structure of Paddle-Inference-Demo/c++/lib as following after copy
Paddle-Inference-Demo/c++/lib/
├── CMakeLists.txt
└── paddle_inference
    ├── CMakeCache.txt
    ├── paddle
    ├── third_party
    └── version.txt

# 3. go to resnet50 demo directory, and download inference model
cd Paddle-Inference-Demo/c++/cpu/resnet50/
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 4. Modify resnet50_test.cc, use config.EnableCustomDevice("npu", 0) to replace config.EnableUseGpu(100, 0)

# 5. Modify compile.sh based on the version.txt in PaddleInfernce C++ package
WITH_MKL=ON  # Turn OFF if aarch64
WITH_GPU=OFF
WITH_ARM=OFF # Turn ON if aarch64

# 6. execute compile script, and executable binary resnet50_test will be generated into build directory
./compile.sh

# 7. execute inference test
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
# expected output
# I0525 11:07:28.354579 40116 resnet50_test.cc:76] run avg time is 713.049 ms
# I0525 11:07:28.354732 40116 resnet50_test.cc:113] 0 : 8.76171e-29
# I0525 11:07:28.354772 40116 resnet50_test.cc:113] 100 : 8.76171e-29
# ... ...
# I0525 11:07:28.354880 40116 resnet50_test.cc:113] 800 : 3.85244e-25
# I0525 11:07:28.354895 40116 resnet50_test.cc:113] 900 : 8.76171e-29
```

## Environment Variables


| Subject     | Variable Name       | Type   | Description    | Default Value |
| -------- | -------------------------------- | ------ | --------------------------------- | ------------------------------------------------------------ |
| Debug     | CUSTOM_DEVICE_BLACK_LIST| String | Ops in back list will fallbacks to CPU  |  ""  |
| Debug     | FLAGS_npu_check_nan_inf | Bool   | check nan or inf of all npu kernels | False                                                       |
| Debug     | FLAGS_npu_blocking_run | Bool   | enable sync for all npu kernels | False                                                     |
| Profiling | FLAGS_npu_profiling_dir | String |   ACL profiling output dir     | "ascend_profiling"                                           |
| Profiling | FLAGS_npu_profiling_dtypes | Uint64 | ACL datatypes to profile | Refer to [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L31) |
| Profiling | FLAGS_npu_profiling_metrics | Uint64 | AI Core metric to profile  | Refer to [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L36) |
| Performance | FLAGS_npu_storage_format         | Bool   | enable Conv/BN acceleration | False                                                        |
