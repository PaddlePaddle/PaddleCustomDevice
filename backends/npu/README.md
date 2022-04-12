# 飞桨自定义接入硬件后端(昇腾NPU)

请参考以下步骤进行硬件后端(昇腾NPU)的编译安装与验证

## 一、环境准备

```bash
# 编译之前需要先保证安装适合昇腾NPU后端使用的Paddle WHL包，当前支持通过源码编译方式获取

# Paddle源码
git clone https://github.com/PaddlePaddle/Paddle.git

# X86_64环境编译
cmake .. -DPY_VERSION=3.7 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
-DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_CUSTOM_DEVICE=ON -DWITH_MKL=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0

make -j$(nproc)

# Aarch64环境编译
pip install patchelf # Paddle内部使用patchelf来修改动态库的rpath
cmake .. -DPY_VERSION=3.7 -DWITH_ARM=ON -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
-DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_CUSTOM_DEVICE=ON -DWITH_XBYAK=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0

make TARGET=ARMV8 -j$(nproc)

# 编译产出在python/dist路径下，使用pip安装
pip install python/dist/paddlepaddle*.whl
```
Paddle源码编译可参考官网：[Linux下从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html#anchor-0)
Aarch64环境编译可参考官网：[飞腾/鲲鹏下从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/arm-compile.html)

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
['Ascend910']
```
