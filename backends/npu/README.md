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

# 2. refer to the following commands to start docker container
docker run -it --name paddle-npu-dev -v `pwd`:/workspace \
       --workdir=/workspace --pids-limit 409600 \
       --privileged --network=host --shm-size=128G \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
       -v /usr/local/dcmi:/usr/local/dcmi \
       registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-$(uname -m)-gcc82 /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle Installation and Verification

> Note: PaddlePaddle Python WHL package supports both training and inference, while ONLY PaddleInference Python API is supported. Please refer to next section if PaddleInference C++ API is needed.

### Source Code Compile

```bash
# 1. go to ascend npu directory
cd backends/npu

# 2. please ensure the PaddlePaddle cpu whl package is already installed
# the development docker image NOT have PaddlePaddle cpu whl installed by default
# you may download and install the nightly built cpu whl package with links below
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_aarch64.whl

# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script - submodules will be synced on demand when compile
bash tools/compile.sh

# 5. install the generated whl package, which is under build/dist directory
pip install build/dist/paddle_custom_npu*.whl
```

## Verification

```bash
# 1. list available custom backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# expected output
['npu']

# 2. check installed custom npu version
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# expected output
version: 0.0.0
commit: d354e1ba347612fe68447e8530d3cd1a0f8aaba9
cann: 6.0.1

# 3. demo for training, evaluation and inference
python tests/test_LeNet_MNIST.py
# expected output - training
Epoch [1/2], Iter [01/14], reader_cost: 2.27062 s, batch_cost: 14.45539 s, ips: 283.35449 samples/s, eta: 0:06:44
Epoch [1/2], Iter [02/14], reader_cost: 1.13547 s, batch_cost: 7.23942 s, ips: 565.79091 samples/s, eta: 0:03:15
... ...
Epoch [2/2], Iter [10/14], reader_cost: 0.24073 s, batch_cost: 0.26355 s, ips: 15541.84990 samples/s, eta: 0:00:01
Epoch [2/2], Iter [11/14], reader_cost: 0.21886 s, batch_cost: 0.24141 s, ips: 16967.21446 samples/s, eta: 0:00:00
Epoch [2/2], Iter [12/14], reader_cost: 0.20063 s, batch_cost: 0.22291 s, ips: 18374.78776 samples/s, eta: 0:00:00
Epoch [2/2], Iter [13/14], reader_cost: 0.18521 s, batch_cost: 0.20728 s, ips: 19760.84536 samples/s, eta: 0:00:00
Epoch [2/2], Iter [14/14], reader_cost: 0.17199 s, batch_cost: 0.19436 s, ips: 21074.31905 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 3.68077 s, reader_cost: 2.40789 s, batch_cost: 2.72104 s, avg ips: 15579.36234 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.86450, Top5 accurary:: 0.99023
# expected output - inference
I0418 16:45:47.717545 85550 interpretercore.cc:267] New Executor is Running.
I0418 16:45:47.788849 85550 analysis_predictor.cc:1414] CustomDevice is enabled
--- Running analysis [ir_graph_build_pass]
I0418 16:45:47.790328 85550 executor.cc:186] Old Executor is Running.
--- Running analysis [ir_analysis_pass]
I0418 16:45:47.792423 85550 ir_analysis_pass.cc:53] argument has no fuse statis
--- Running analysis [ir_params_sync_among_devices_pass]
I0418 16:45:47.792572 85550 ir_params_sync_among_devices_pass.cc:142] Sync params from CPU to CustomDevicenpu/0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0418 16:45:47.880336 85550 analysis_predictor.cc:1565] ======= optimize end =======
I0418 16:45:47.880510 85550 naive_executor.cc:151] ---  skip [feed], feed -> inputs
I0418 16:45:47.881462 85550 naive_executor.cc:151] ---  skip [linear_5.tmp_1], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
```

## PaddleInference C++ Installation and Verification

### PaddleInference C++ Source Compile

> Note: the official released PaddleInference C++ package do not support custom device, please follow the steps below to source compile PaddleInference C++ package.

```bash
# 1. got to Paddle source code directory
cd PaddleCustomDevice/Paddle

# 2. prepare build directory
mkdir build && cd build

# 3.1 build command for X86_64
cmake .. -DPY_VERSION=3.9 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_ARM=OFF
make -j8

# 3.2 build command for aarch64
cmake .. -DPY_VERSION=3.9 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_ARM=ON
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
