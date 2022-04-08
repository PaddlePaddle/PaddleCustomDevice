# paddle_ascend

飞桨支持昇腾NPU芯片的插件

## 源码同步
```
git clone https://github.com/PaddlePaddle/PaddleCustomDevice.git

git submodule
-8fb8fa4109592c49b995be9b246c30d40bce6935 Paddle

git submodule init
Submodule 'Paddle' (https://github.com/PaddlePaddle/Paddle) registered for path 'Paddle'

git submodule update
Cloning into '/workspace/PaddleCustomDevice/Paddle'...
Submodule path 'Paddle': checked out '8fb8fa4109592c49b995be9b246c30d40bce6935'
```
或使用：
```
git clone https://github.com/PaddlePaddle/PaddleCustomDevice.git --recursive

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
