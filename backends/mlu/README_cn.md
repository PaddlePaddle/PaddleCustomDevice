# 飞桨自定义接入硬件后端(寒武纪MLU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(寒武纪MLU)的编译安装与验证

## 依赖模块版本信息

| 模块名称  | 版本     |
| --------- | -------- |
| cntoolkit | 3.1.2-1  |
| cnnl      | 1.13.2-1 |
| cncl      | 1.4.1-1  |
| mluops    | 0.3.0-1  |

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
# 进入硬件后端(寒武纪MLU)目录
cd backends/mlu

# 编译之前需要先保证环境下装有Paddle WHL包，可以直接安装CPU版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 创建编译目录并编译
mkdir build && cd build

# X86_64环境编译
cmake ..
make -j8

# Aarch64环境编译
cmake .. -DWITH_ARM=ON
make TARGET=ARMV8 -j8

# 编译产出在dist路径下，使用pip安装
pip install dist/paddle_custom_mlu*.whl
```

## 三、功能验证

```bash
# 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 期待输出以下结果
['CustomMLU']

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
