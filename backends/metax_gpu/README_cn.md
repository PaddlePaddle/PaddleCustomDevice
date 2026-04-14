# 飞桨自定义接入硬件后端(沐曦GPU)

简体中文 | [English](./README.md)

请参考以下步骤进行编译安装与验证

## 安装paddle-cpu
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

## 编译安装

```bash
# 获取沐曦PaddlePaddle Docker镜像

# 克隆PaddleCustomDevice源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 编译安装
cd backends/metax_gpu
bash build_in_custom.sh
# 或者
bash change_patch.sh #只执行一次
bash compile.sh      # 可执行多次
```

## 验证

```bash

# 运行测试
cd tests
bash run_test.sh
```
