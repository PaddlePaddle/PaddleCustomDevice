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
pip install paddlepaddle==2.5.0 -f https://paddle-device.bj.bcebos.com/2.5.0/cpu/paddlepaddle-2.5.0-cp37-cp37m-linux_x86_64.whl

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
['mlu']

# 2) 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: 7112037a5b7149bc165e8008fb70c72ba71beb04

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


## 四、PaddleInference的安装和验证
### 4.1 PaddleInference C++ 预测库编译
> 注意：飞桨官网发布的 PaddleInference C++ 预测库中默认不含有 CustomDevice 功能支持，因此这里我们需要重新编译得到 PaddleInference C++ 预测库。

```shell
# 1）进入 PaddlePaddle 主框架源码目录
cd PaddleCustomDevice/Paddle

# 2）创建编译目录
mkdir build && cd build

# 3.1）X86-64 环境下的编译命令 - 编译 CPU 版本即可
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_ARM=OFF
make -j8

# 3.2) Aarch64 环境下的编译命令 - 编译 CPU 版本即可
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_ARM=ON
make TARGET=ARMV8 -j8

# 4) 生成的 PaddleInference C++ 预测库即为 build/paddle_inference_install_dir 目录
```

### MLU CustomDevice compile
```shell
# 1) 安装PaddlePaddle-CPU
pip install --force-reinstall python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl

# 2) 进入 PaddleCustomDevice 的 MLU 安装目录
cd backends/mlu

# 3）设置 PaddleInference 库的环境变量
export PADDLE_INFERENCE_LIB_DIR=/path/to/Paddle/build/paddle_inference_install_dir/paddle/lib

# 4）编译 paddle-custom-mlu
mkdir build && cd build
cmake .. -DWITH_TESTING=ON -DON_INFER=ON
make -j32

# 5）安装 paddle-custom-mlu
pip install --force-reinstall dist/paddle_custom_mlu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 6）编译产出为 build 目录下的 libpaddle-custom-npu.so 文件，指定插件路径到库文件目录下
export CUSTOM_DEVICE_ROOT=/path/to/PaddleCustomDevice/backends/mlu/build
```

### Run resnet50 with paddleinference
```shell
# 1) 下载 Paddle-Inference-Demo 代码
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git

# 2) 拷贝源码编译生成的 C++ 预测库到 Paddle-Inference-Demo/c++/lib 目录下
cp -r PaddleCustomDevice/Paddle/build/paddle_inference_install_dir Paddle-Inference-Demo/c++/lib/paddle_inference
# 拷贝完成之后 Paddle-Inference-Demo/c++/lib 目录结构如下
Paddle-Inference-Demo/c++/lib/
├── CMakeLists.txt
└── paddle_inference
    ├── CMakeCache.txt
    ├── paddle
    ├── third_party
    └── version.txt

# 3) 进入 C++ 示例代码目录，下载推理模型
cd Paddle-Inference-Demo/c++/cpu/resnet50/
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 4) 修改 resnet50_test.cc，使用 mlu 作为设备传入
config.EnableCustomDevice("mlu", 0);

# 5) 修改 compile.sh 编译文件，需根据 C++ 预测库的 version.txt 信息对以下的几处内容进行修改
WITH_MKL=ON  # Turn OFF if aarch64
WITH_GPU=OFF
WITH_ARM=OFF # Turn ON if aarch64

# 6) 执行编译，编译完成之后在 build 下生成 resnet50_test 可执行文件
./compile.sh

# 7) 运行 C++ 预测程序
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --repeats 1000
# 预期得到如下输出结果
I0216 03:11:44.346031 17725 resnet50_test.cc:77] run avg time is 12.0345 ms
I0216 03:11:44.346101 17725 resnet50_test.cc:92] 0 : 2.71852e-43
I0216 03:11:44.346136 17725 resnet50_test.cc:92] 100 : 2.04159e-37
I0216 03:11:44.346144 17725 resnet50_test.cc:92] 200 : 2.12377e-33
I0216 03:11:44.346151 17725 resnet50_test.cc:92] 300 : 5.16799e-42
I0216 03:11:44.346158 17725 resnet50_test.cc:92] 400 : 1.68488e-35
I0216 03:11:44.346164 17725 resnet50_test.cc:92] 500 : 7.00649e-45
I0216 03:11:44.346171 17725 resnet50_test.cc:92] 600 : 1.05766e-19
I0216 03:11:44.346176 17725 resnet50_test.cc:92] 700 : 2.04091e-23
I0216 03:11:44.346184 17725 resnet50_test.cc:92] 800 : 3.85242e-25
I0216 03:11:44.346190 17725 resnet50_test.cc:92] 900 : 1.52387e-30
```

## 四、环境变量

### PADDLE_MLU_ALLOW_TF32
该功能使Conv，MatMul类算子以TF32数据类型进行计算，目前只支持MLU590板卡，TF32是MLU590运行的默认数据类型。

开启TF32数据类型计算。
```bash
export PADDLE_MLU_ALLOW_TF32=true
```
