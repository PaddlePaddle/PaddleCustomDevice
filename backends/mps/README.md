#PaddlePaddle Custom Device Implementaion for Custom CPU


Please refer to the following steps to compile, install and verify the custom device implementaion for MPS backend.

## Prepare environment and source code

```bash
# 1. clone the source code recursively along with Paddle source code
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 2. execute the following commands to update submodule
git submodule sync
git submodule update --remote --init --recursive
```

## Compile and Install

```bash
#navigate to implementaion for MPS backend.
cd backends/mps

#before compiling, ensure that Paddle is installed, you can run the following command
pip install paddlepaddle
#create the build directory and navigate in
mkdir build && cd build

#Currently, a SIGN_IDENTITY is required to sign the dynamic library(.dylib)
cmake ..-D SIGN_IDENTITY=<Your Identity>
make -j8

#using pip to install the output
pip install dist/paddle_mps*.whl
```

## Verification

```bash
#list available hardware backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"

#expected output
['mps']

#run a simple model
python -c "import paddle; paddle.set_device('mps'); print(paddle.nn.functional.softmax(paddle.ones([2])))"

#expected similar output
... ...
Tensor(shape=[2], dtype=float32, place=Place(mps:0), stop_gradient=True,
       [0.50000000, 0.50000000])
```
