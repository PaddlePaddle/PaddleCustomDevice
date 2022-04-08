# paddle_ascend

飞桨支持昇腾NPU芯片的插件

## 源码同步
```
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive --jobs 0
```

## 编译安装
```
cd ascend
mkdir build && cd build
cmake .. -DWITH_KERNELS=ON # 如果是ARM环境添加 -DWITH_ARM=ON
make

# 编译产出在dist路径下，使用pip安装
pip install dist/paddle_ascend*.whl

```
