# 飞桨自定义接入硬件后端(昇腾NPU)

请参考以下步骤进行硬件后端(昇腾NPU)的编译安装与验证

## 一、源码同步

```bash
# 克隆代码
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 请执行以下命令，以保证checkout最新的Paddle源码
git submodule sync
git submodule update --remote --init --recursive
```

## 二、编译安装

```bash
# 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 编译之前需要先保证环境下装有Paddle WHL包，可以直接安装CPU版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

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

## 三、功能验证

```bash
# 列出可用硬件后端
python -c "import paddle; paddle.device.get_all_custom_device_type()"
# 期待输出以下结果
['Ascend910']
```
