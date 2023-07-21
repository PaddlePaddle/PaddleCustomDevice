# PaddlePaddle Custom Device Implementaion for Cambricon MLU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementaion for Cambricon MLU.

## Neuware Version

| Module    | Version  |
| --------- | -------- |
| cntoolkit | 3.4.2-1  |
| cnnl      | 1.17.0-1 |
| cncl      | 1.9.3-1  |
| mluops    | 0.6.0-1  |

## Prepare environment and source code

```bash
# 1. pull PaddlePaddle Cambricon MLU development docker image
# dockerfile of the image is in tools/dockerfile directory
docker pull registry.baidubce.com/device/paddle-mlu:cntoolkit3.4.2-1-cnnl1.17.0-1-gcc82

# 2. refer to the following commands to start docker container
docker run -it --name paddle-mlu-dev -v `pwd`:/workspace \
    --shm-size=128G --network=host -w=/workspace \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --privileged -v /usr/bin/cnmon:/usr/bin/cnmon \
    registry.baidubce.com/device/paddle-mlu:cntoolkit3.4.2-1-cnnl1.17.0-1-gcc82 /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle Installation and Verification

> Note: PaddlePaddle Python WHL package supports both training and inference, while ONLY PaddleInference Python API is supported. Please refer to next section if PaddleInference C++ API is needed.

### Source Code Compilation

```bash
# 1. navigate to implementaion for Cambricon MLU
cd backends/mlu

# 2. before compiling, ensure that PaddlePaddle (CPU version) is installed, you can run the following command
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script - submodules will be synced on demand when compile
bash tools/compile.sh

# 5. install the generated whl package, which is under build/dist directory
pip install build/dist/paddle_custom_mlu*.whl
```

## Verification

```bash
# list available custom backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"

# expected output
['mlu']

# 2. check installed custom mlu version
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# expected output
version: 0.0.0
commit: 98ae7a84b51e36fc15a1fef7808a82ed8b792fdf
cntoolkit: 3.4.2
cnnl: 1.17.0
cncl: 1.9.3
mluops: 0.6.0

# 3. demo for training, evaluation and inference
python tests/test_LeNet_MNIST.py
# expected output - training
Epoch [1/2], Iter [01/14], reader_cost: 1.23499 s, batch_cost: 1.28666 s, ips: 3183.43983 samples/s, eta: 0:00:36
Epoch [1/2], Iter [02/14], reader_cost: 0.61760 s, batch_cost: 0.66744 s, ips: 6136.88837 samples/s, eta: 0:00:18
... ...
Epoch [2/2], Iter [10/14], reader_cost: 0.12527 s, batch_cost: 0.17382 s, ips: 23565.25098 samples/s, eta: 0:00:00
Epoch [2/2], Iter [11/14], reader_cost: 0.11389 s, batch_cost: 0.16232 s, ips: 25233.58550 samples/s, eta: 0:00:00
Epoch [2/2], Iter [12/14], reader_cost: 0.10441 s, batch_cost: 0.15278 s, ips: 26810.42007 samples/s, eta: 0:00:00
Epoch [2/2], Iter [13/14], reader_cost: 0.09639 s, batch_cost: 0.14467 s, ips: 28312.30064 samples/s, eta: 0:00:00
Epoch [2/2], Iter [14/14], reader_cost: 0.08951 s, batch_cost: 0.13775 s, ips: 29735.07966 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 1.96801 s, reader_cost: 1.25314 s, batch_cost: 1.92850 s, avg ips: 29138.02172 samples/s
# expected output - inference
I0705 17:13:54.706485 42228 program_interpreter.cc:171] New Executor is Running.
I0705 17:13:54.730235 42228 analysis_predictor.cc:1502] CustomDevice is enabled
--- Running analysis [ir_graph_build_pass]
I0705 17:13:54.730543 42228 executor.cc:187] Old Executor is Running.
--- Running analysis [ir_analysis_pass]
I0705 17:13:54.730988 42228 ir_analysis_pass.cc:46] argument has no fuse statis
--- Running analysis [save_optimized_model_pass]
W0705 17:13:54.731001 42228 save_optimized_model_pass.cc:28] save_optim_cache_model is turned off, skip save_optimized_model_pass
--- Running analysis [ir_params_sync_among_devices_pass]
I0705 17:13:54.731010 42228 ir_params_sync_among_devices_pass.cc:142] Sync params from CPU to mlu:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0705 17:13:54.733326 42228 analysis_predictor.cc:1676] ======= optimize end =======
I0705 17:13:54.733381 42228 naive_executor.cc:171] ---  skip [feed], feed -> inputs
I0705 17:13:54.733578 42228 naive_executor.cc:171] ---  skip [linear_5.tmp_1], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
```

## PaddleInference C++ Installation and Verification

### PaddleInference C++ Source Compile

> Note: the official released PaddleInference C++ package do not support custom device, please follow the steps below to source compile PaddleInference C++ package.

```bash
# 1. go to Paddle source code directory
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

### Cambricon MLU Inference Source Compile
```bash
# 1. go to cambricon mlu directory
cd backends/mlu

# 2. compile options, the PADDLE_INFERENCE_LIB_DIR is the path of Paddle Inference C++ package
# generated in the previous step, i.e. build/paddle_inference_install_dir directory
export ON_INFER=ON # whether to enable C++ inference, default is OFF
export PADDLE_INFERENCE_LIB_DIR=/path/to/Paddle/build/paddle_inference_install_dir

# 3. execute compile script
bash tools/compile.sh

# 4. Specify CUSTOM_DEVICE_ROOT to the folder of libpaddle-custom-mlu.so
export CUSTOM_DEVICE_ROOT=/path/to/PaddleCustomDevice/backends/mlu/build
```

### Cambricon MLU Inference Verification

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

# 4. Modify resnet50_test.cc, use use mlu as devcie in config
config.EnableCustomDevice("mlu", 0);

# 5. Modify compile.sh based on the version.txt in PaddleInfernce C++ package
WITH_MKL=ON  # Turn OFF if aarch64
WITH_GPU=OFF
WITH_ARM=OFF # Turn ON if aarch64

# 6. execute compile script, and executable binary resnet50_test will be generated into build directory
./compile.sh

# 7. execute inference test
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --repeats 1000
# expected output
I0216 03:11:44.346031 17725 resnet50_test.cc:77] run avg time is 12.0345 ms
I0216 03:11:44.346101 17725 resnet50_test.cc:92] 0 : 2.71852e-43
I0216 03:11:44.346136 17725 resnet50_test.cc:92] 100 : 2.04159e-37
I0216 03:11:44.346144 17725 resnet50_test.cc:92] 200 : 2.12377e-33
I0216 03:11:44.346151 17725 resnet50_test.cc:92] 300 : 5.16799e-42
I0216 03:11:44.346158 17725 resnet50_test.cc:92] 400 : 1.68488e-35
I0216 03:11:44.346164 17725 resnet50_test.cc:92] 500 : 7.00649e-45
I0216 03:11:44.346171 17725 resnet50_test.cc:92] 600 : 1.05766e-19
I0216 03:11:44.346176 17725 resnet50_test.cc:92] 700 : 2.04091e-23
I0216 03:11:44.346184 17725 resnet50_test.cc:92] 800 : 3.85242e-25
I0216 03:11:44.346190 17725 resnet50_test.cc:92] 900 : 1.52387e-30
```

## Environment variables

### PADDLE_MLU_ALLOW_TF32
This function enables Conv, MatMul operators to be computed with TF32 data type. Currently, only MLU590 are supported, and TF32 is the default data type for MLU590 operation.

Turn on TF32 data type calculation.
```bash
export PADDLE_MLU_ALLOW_TF32=true
```

### CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE
This function controls whether to enable multiply clique of CNCL. Currently, CNCL will not accept multiply communication clique by default, which will cause process stuck in some multi-clique collective communication senario.

Turn off multi-clique mem restriction.
```bash
export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1
```
