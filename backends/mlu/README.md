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
## Environment variables

### PADDLE_MLU_ALLOW_TF32
This function enables Conv, MatMul operators to be computed with TF32 data type. Currently, only MLU590 are supported, and TF32 is the default data type for MLU590 operation.

Turn on TF32 data type calculation.
```bash
export PADDLE_MLU_ALLOW_TF32=true
```
