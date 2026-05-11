# 飞桨自定义接入硬件后端（Intel GPU）

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端（Intel GPU）的编译安装与验证

## 激活 Intel oneAPI 环境变量

```bash
source load.sh
```

## 获取源码

```bash
# 克隆代码
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 请执行以下命令，以保证checkout最新的Paddle源码
git submodule sync
git submodule update --remote --init --recursive
```

## Compile and Install

```bash
# 进入 Intel GPU Backend 目录
cd backends/intel_gpu

# 编译前，请确保已安装 Paddle，您可以运行以下命令
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 创建编译目录并编译
mkdir build && cd build

cmake ..
make -j $(nproc)

# 编译产出在dist路径下，使用pip安装
pip install dist/paddle_custom_intel_gpu*.whl
```

## 验证

```bash
# 检查插件状态
python -c "import paddle; print('intel_gpu' in paddle.device.get_all_custom_device_type())"

# 预期输出
True

```
