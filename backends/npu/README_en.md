# PaddlePaddle Custom Device Implementaion for Ascend NPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementaion for Ascend NPU.

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
# go to ascend npu directory
cd backends/npu

# install paddlepaddle, propose to use cpu version
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# compile options, whether to compile with unit testing, default is OFF
export WITH_TESTING=OFF

# execute compile script
bash tools/compile.sh

# generated package is under build/dist directory
pip install build/dist/paddle_custom_npu*.whl
```

## Verification

```bash
# list available hardware backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"

# expected output
['npu']

# run a simple model
python tests/test_MNIST_model.py

# expected similar output 
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

## Using PaddleInference

Re-compile plugin

```bash
# Compile PaddleInference
git clone https://github.com/PaddlePaddle/Paddle.git
git clone https://github.com/ronny1996/Paddle-Inference-Demo.git

mkdir -p Paddle/build
pushd Paddle/build

cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_ARM=ON -DWITH_ASCEND=OFF -DWITH_ASCEND_CL=ON -DWITH_TESTING=ON -DWITH_DISTRIBUTE=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DPYTHON_INCLUDE_DIR=`python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"` -DWITH_CUSTOM_DEVICE=ON -DWITH_ASCEND_CXX11=ON

make TARGET=ARMV8 -j8 # or make -j8

popd
cp -R Paddle/build/paddle_inference_install_dir Paddle-Inference-Demo/c++/lib/paddle_inference
export PADDLE_INFERENCE_LIB_DIR=$(realpath Paddle-Inference-Demo/c++/lib/paddle_inference/paddle/lib)

# Compile the plug-in
mkdir -p PaddleCustomDevice/backends/npu/build
pushd PaddleCustomDevice/backends/npu/build

# Compile in X86_64 environment
cmake .. -DON_INFER=ON -DPADDLE_INFERENCE_LIB_DIR=${PADDLE_INFERENCE_LIB_DIR}
make -j8

# Compile in Aarch64 environment
cmake .. -DWITH_ARM=ON -DON_INFER=ON -DPADDLE_INFERENCE_LIB_DIR=${PADDLE_INFERENCE_LIB_DIR}
make TARGET=ARMV8 -j8

# Specify the plug-in directory
export CUSTOM_DEVICE_ROOT=$PWD
popd
```

Using PaddleInference

```bash
pushd Paddle-Inference-Demo/c++/resnet50

# Modify resnet50_test.cc, use config.EnableCustomDevice("npu", 0) to replace config.EnableUseGpu(100, 0)
  
bash run.sh
```

expected similar output 

```bash
I0516 14:40:56.197255 114531 resnet50_test.cc:74] run avg time is 115421 ms
I0516 14:40:56.197389 114531 resnet50_test.cc:89] 0 : 2.67648e-43
I0516 14:40:56.197425 114531 resnet50_test.cc:89] 100 : 1.98479e-37
I0516 14:40:56.197445 114531 resnet50_test.cc:89] 200 : 2.05547e-33
I0516 14:40:56.197463 114531 resnet50_test.cc:89] 300 : 5.06149e-42
I0516 14:40:56.197474 114531 resnet50_test.cc:89] 400 : 1.58719e-35
I0516 14:40:56.197484 114531 resnet50_test.cc:89] 500 : 7.00649e-45
I0516 14:40:56.197494 114531 resnet50_test.cc:89] 600 : 1.00972e-19
I0516 14:40:56.197504 114531 resnet50_test.cc:89] 700 : 1.92904e-23
I0516 14:40:56.197512 114531 resnet50_test.cc:89] 800 : 3.80365e-25
I0516 14:40:56.197522 114531 resnet50_test.cc:89] 900 : 1.46266e-30
```
