# 飞桨自定义接入硬件后端(昇腾NPU)

请参考以下步骤进行硬件后端(昇腾NPU)的编译安装与验证

## 一、环境准备

编译与验证前需要先安装适合昇腾NPU后端使用的Paddle WHL包，当前支持通过源码编译方式获取

```bash
# 克隆代码
git clone https://github.com/PaddlePaddle/Paddle.git

# 进入Paddle目录
cd Paddle

# 创建编译目录并编译
mkdir build && cd build

# X86_64环境编译
cmake .. -DPY_VERSION=3.7 -DWITH_CUSTOM_DEVICE=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
-DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_PSCORE=OFF -DWITH_MKL=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
make -j$(nproc)

# Aarch64环境编译
pip install patchelf # Paddle内部使用patchelf来修改动态库的rpath
cmake .. -DPY_VERSION=3.7 -DWITH_ARM=ON -DWITH_CUSTOM_DEVICE=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
-DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_PSCORE=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
make TARGET=ARMV8 -j$(nproc)

# 编译产出在python/dist路径下，使用pip安装
pip install python/dist/paddlepaddle*.whl
```

## 二、源码同步

```bash
# 克隆代码
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 请执行以下命令，以保证checkout最新的Paddle源码
git submodule sync
git submodule update --remote --init --recursive
```

## 三、编译安装

```bash
# 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 创建编译目录并编译
mkdir build && cd build

# X86_64环境编译
cmake .. -DWITH_KERNELS=ON
make -j8

# Aarch64环境编译
cmake .. -DWITH_KERNELS=ON -DWITH_ARM=ON
make TARGET=ARMV8 -j8

# 编译产出在dist路径下，使用pip安装
pip install dist/paddle_custom_npu*.whl
```

## 四、功能验证

```bash
# 列出可用硬件后端
python -c "import paddle; paddle.device.get_all_custom_device_type()"
# 期待输出以下结果
['ascend']
# 运行简单模型
python ../tests/test_MNIST_model.py 
# 期待输出以下类似结果
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
