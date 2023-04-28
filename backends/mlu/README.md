# PaddlePaddle Custom Device Implementaion for Cambricon MLU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementaion for Cambricon MLU.

## Dependencies Version

| Module    | Version  |
| --------- | -------- |
| cntoolkit | 3.1.2-1  |
| cnnl      | 1.13.2-1 |
| cncl      | 1.4.1-1  |
| mluops    | 0.3.0-1  |

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
# navigate to implementaion for Cambricon MLU
cd backends/mlu

# before compiling, ensure that Paddle is installed, you can run the following command
pip install paddlepaddle==2.5.0 -f https://paddle-device.bj.bcebos.com/2.5.0/cpu/paddlepaddle-2.5.0-cp37-cp37m-linux_x86_64.whl

# create the build directory and navigate in
mkdir build && cd build

# compile in X86_64 environment
cmake ..
make -j8

# using pip to install the output
pip install dist/paddle_custom_mlu*.whl
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
commit: 7112037a5b7149bc165e8008fb70c72ba71beb04

# run a simple model
python ../tests/test_MNIST_model.py

# expected similar output
... ...
Epoch 0 step 0, Loss = [2.2901897], Accuracy = [0.140625]
Epoch 0 step 100, Loss = [1.7438297], Accuracy = [0.78125]
Epoch 0 step 200, Loss = [1.6934999], Accuracy = [0.796875]
Epoch 0 step 300, Loss = [1.6888921], Accuracy = [0.78125]
Epoch 0 step 400, Loss = [1.7731808], Accuracy = [0.734375]
Epoch 0 step 500, Loss = [1.7497146], Accuracy = [0.71875]
Epoch 0 step 600, Loss = [1.5952139], Accuracy = [0.875]
Epoch 0 step 700, Loss = [1.6935768], Accuracy = [0.78125]
Epoch 0 step 800, Loss = [1.695106], Accuracy = [0.796875]
Epoch 0 step 900, Loss = [1.6372337], Accuracy = [0.828125]
```

## PaddleInference C++ Installation and Verification
### PaddleInference C++ Source Compile
> Note: the official released PaddleInference C++ package do not support custom device, please follow the steps below to source compile PaddleInference C++ package.

```shell
# 1. got to Paddle source code directory under PaddleCustomDevice
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

### MLU CustomDevice compile
```shell
# 1. install paddlepaddle-cpu
pip install --force-reinstall python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl

# 2. go to mlu directory under PaddleCustomDevice
cd backends/mlu

# 3. set ENV variable of paddleinference dir
export PADDLE_INFERENCE_LIB_DIR=/path/to/Paddle/build/paddle_inference_install_dir/paddle/lib

# 4. build paddle-custom-mlu
mkdir build && cd build
cmake .. -DWITH_TESTING=ON -DON_INFER=ON
make -j32

# 5. install paddle-custom-mlu
pip install --force-reinstall dist/paddle_custom_mlu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 6. set ENV CUSTOM_DEVICE_ROOT
export CUSTOM_DEVICE_ROOT=/path/to/PaddleCustomDevice/backends/mlu/build
```

### Run resnet50 with paddleinference
```shell
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
