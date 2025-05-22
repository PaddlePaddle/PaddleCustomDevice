# 飞桨自定义接入硬件后端(天数GPU)

简体中文 | [English](./README.md)

请参考以下步骤进行编译安装与验证

## 编译安装

```bash
# 获取天数PaddlePaddle Docker镜像

# 克隆PaddleCustomDevice源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 编译安装
cd backends/iluvatar_gpu
bash build_paddle.sh
```

## 验证

```bash
# build with BUILD_TEST=1

# run_test
cd tests
mkdir -p build && cd build && cmake ..
make run_test
```
