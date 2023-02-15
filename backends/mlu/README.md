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
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

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
# list available hardware backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"

# expected output
['CustomMLU']

# run a simple model
python ../tests/test_MNIST_model.py

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
## Environment variables

### PADDLE_MLU_ALLOW_TF32
This function enables Conv, MatMul operators to be computed with TF32 data type. Currently, only MLU590 are supported, and TF32 is the default data type for MLU590 operation.

Turn on TF32 data type calculation.
```bash
export PADDLE_MLU_ALLOW_TF32=true
```
