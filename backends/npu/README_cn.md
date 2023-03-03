# 飞桨自定义接入硬件后端(昇腾NPU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(昇腾NPU)的编译安装与验证

## 环境准备与源码同步

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与 dockerfile 位于 tools/dockerfile 目录下
docker pull registry.baidubce.com/device/paddle-npu:cann631-x86_64-gcc82
docker pull registry.baidubce.com/device/paddle-npu:cann631-aarch64-gcc82

# 2) 参考如下命令启动容器
docker run -it --name paddle-dev-cann600 -v `pwd`:/workspace \
       --workdir=/workspace --pids-limit 409600 \
       --privileged --network=host --shm-size=128G \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
       -v /usr/local/dcmi:/usr/local/dcmi \
       registry.baidubce.com/device/paddle-npu:cann631-$(uname -m)-gcc82 /bin/bash

# 3) 克隆源码，注意 PaddleCustomDevice 依赖 PaddlePaddle 主框架源码
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 4) 请执行以下命令，以保证 checkout 最新的 PaddlePaddle 主框架源码
git submodule sync
git submodule update --remote --init --recursive
```

## PaddlePaddle 训练安装与运行

### 训练编译安装

```bash
# 1) 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 2) 编译之前需要先保证环境下装有飞桨安装包，直接安装飞桨 CPU 版本即可
# 默认 NPU 开发镜像中已经装有飞桨 CPU 安装包 (飞桨 develop 分支的 nightly build 版本)
# 也可以通过如下地址下载得到 PaddlePaddle develop 分支的 nightly build 版本的安装包
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp37-cp37m-linux_aarch64.whl

# 3) 编译选项，是否打开单元测试编译，默认值为 ON
export WITH_TESTING=OFF

# 4) 执行编译脚本
bash tools/compile.sh

# 5) 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/paddle_custom_npu*.whl
```

### 训练功能验证

```bash
# 1) 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果
['npu']

# 2) 运行简单模型训练任务
python tests/test_MNIST_model.py
# 预期得到如下输出结果
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

## PaddleInference 推理安装与运行

### PaddleInference C++ 预测库编译

> 注意：飞桨官网发布的 PaddleInference C++ 预测库中默认不含有 CustomDevice 功能支持，因此这里我们需要重新编译得到 PaddleInference C++ 预测库。

```bash
# 1) 进入 PaddlePaddle 主框架源码目录
cd PaddleCustomDevice/Paddle

# 2) 创建编译目录
mkdir build && cd build

# 3.1) X86-64 环境下的编译命令 - 编译 CPU 版本即可
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_ARM=OFF
make -j8

# 3.2) Aarch64 环境下的编译命令 - 编译 CPU 版本即可
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_ARM=ON
make TARGET=ARMV8 -j8

# 4) 生成的 PaddleInference C++ 预测库即为 build/paddle_inference_install_dir 目录
```

### 推理编译安装

```bash
# 1) 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 2) 编译选项，PADDLE_INFERENCE_LIB_DIR 为上一步编译得到的 C++ 预测库的地址
export ON_INFER=ON # 是否打开推理库编译，默认为 OFF
export PADDLE_INFERENCE_LIB_DIR=/path/to/Paddle/build/paddle_inference_install_dir

# 3) 执行编译脚本
bash tools/compile.sh

# 4) 编译产出为 build 目录下的 libpaddle-custom-npu.so 文件，指定插件路径到库文件目录下
export CUSTOM_DEVICE_ROOT=/path/to/PaddleCustomDevice/backends/npu/build
```

### 推理功能验证

```bash
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

# 4) 修改 resnet50_test.cc，使用 config.EnableCustomDevice("npu", 0) 接口替换 config.EnableUseGpu(100, 0)

# 5) 修改 compile.sh 编译文件，需根据 C++ 预测库的 version.txt 信息对以下的几处内容进行修改
WITH_MKL=ON  # 如果是 Aarch 环境，请设置为 OFF
WITH_GPU=OFF
WITH_ARM=OFF # 如果是 Aarch 环境，请设置为 ON

# 6) 执行编译，编译完成之后在 build 下生成 resnet50_test 可执行文件
./compile.sh

# 7) 运行 C++ 预测程序
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
# 预期得到如下输出结果
# I0525 11:07:28.354579 40116 resnet50_test.cc:76] run avg time is 713.049 ms
# I0525 11:07:28.354732 40116 resnet50_test.cc:113] 0 : 8.76171e-29
# I0525 11:07:28.354772 40116 resnet50_test.cc:113] 100 : 8.76171e-29
# ... ...
# I0525 11:07:28.354880 40116 resnet50_test.cc:113] 800 : 3.85244e-25
# I0525 11:07:28.354895 40116 resnet50_test.cc:113] 900 : 8.76171e-29
```

### 环境变量

| 主题   | 变量名称                         | 类型   | 说明                              | 默认值                                                       |
| -------- | -------------------------------- | ------ | --------------------------------- | ------------------------------------------------------------ |
| 调试     | FLAGS_npu_check_nan_inf | Bool   | 是否开启所有NPU算子输入输出检查   | False                                                        |
| 调试     | FLAGS_npu_blocking_run | Bool   | 是否开启强制同步执行所有 NPU 算子 | False                                                        |
| 性能分析 | FLAGS_npu_profiling_dir | String | 设置 Profiling 数据保存目录       | "ascend_profiling"                                           |
| 性能分析 | FLAGS_npu_profiling_dtypes | Uint64 | 指定需要采集的 Profiling 数据类型 | 见 [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L28) |
| 性能分析 | FLAGS_npu_profiling_metrics | Uint64 | 设置 AI Core 性能指标采集项       | 见 [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L28) |
| 性能加速 | FLAGS_npu_storage_format  | Bool   | 是否开启 Conv/BN 等算子的计算加速 | False                                                        |
