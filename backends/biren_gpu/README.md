# 飞桨自定义接入硬件后端(壁仞GPU)

简体中文 | [English](./README_en.md)

请参考以下步骤进行编译安装与验证

## 编译安装

```bash
# 获取壁仞PaddlePaddle Docker镜像

# 克隆PaddleCustomDevice源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 编译安装
cd backends/biren_gpu
mkdir -p build
pushd build
cmake -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug ..
ninja
pip3 install --no-index --find-links=offline dist/paddle_custom_supa-1.0.0.0*.whl --force-reinstall
```

## 验证

```bash
python3 tests/test_abs_op_supa.py
```
