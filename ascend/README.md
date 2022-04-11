# 飞桨自定义接入硬件后端(昇腾NPU)

## 源码同步

```
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
# 请执行以下命令，以保证checkout最新的Paddle源码
git submodule sync
git submodule update --remote --init --recursive
```

## 编译安装

```
cd ascend
mkdir build && cd build
cmake .. -DWITH_KERNELS=ON # 如果是ARM环境添加 -DWITH_ARM=ON
make -j8

# 编译产出在dist路径下，使用pip安装
pip install dist/paddle_ascend*.whl
```
