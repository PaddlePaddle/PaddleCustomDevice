# 飞桨自定义接入硬件后端(昇腾NPU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(昇腾NPU)的编译安装与验证

## 环境准备与源码同步

> 注意：当前支持 [CANN 6.0.1](https://www.hiascend.com/software/cann/community-history?id=6.0.1.alpha001) 版本

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与 dockerfile 位于 tools/dockerfile 目录下
docker pull registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-x86_64-gcc82
docker pull registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-aarch64-gcc82

# 2) 参考如下命令启动容器
docker run -it --name paddle-npu-dev -v `pwd`:/workspace \
       --workdir=/workspace --pids-limit 409600 \
       --privileged --network=host --shm-size=128G \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
       -v /usr/local/dcmi:/usr/local/dcmi \
       registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-$(uname -m)-gcc82 /bin/bash

# 3) 克隆 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle 安装与运行

> 注意：此步骤编译得到的 PaddlePaddle Python WHL 安装包同时包含训练和推理功能，其中推理仅支持 PaddleInference Python API，如果需要 PaddleInference C++ API 请参考下一章节 "PaddleInference C++ 推理安装与运行"。

### 编译安装

```bash
# 1) 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 2) 编译之前需要先保证环境下装有飞桨安装包，直接安装飞桨 CPU 版本即可
# 默认开发镜像中不含有飞桨安装包，可通过如下地址安装 PaddlePaddle develop 分支的 nightly build 版本的安装包
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl
https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_aarch64.whl

# 3) 编译选项，是否打开单元测试编译，默认值为 ON
export WITH_TESTING=OFF

# 4) 执行编译脚本 - submodule 在编译时会按需下载
bash tools/compile.sh

# 5) 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/paddle_custom_npu*.whl
```

### 功能验证

```bash
# 1) 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果
['npu']

# 2) 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: d354e1ba347612fe68447e8530d3cd1a0f8aaba9
cann: 6.0.1

# 3) 运行简单模型训练、评估和推理任务
python tests/test_LeNet_MNIST.py
# 预期得到如下输出结果 - 训练输出
Epoch [1/2], Iter [01/14], reader_cost: 2.27062 s, batch_cost: 14.45539 s, ips: 283.35449 samples/s, eta: 0:06:44
Epoch [1/2], Iter [02/14], reader_cost: 1.13547 s, batch_cost: 7.23942 s, ips: 565.79091 samples/s, eta: 0:03:15
... ...
Epoch [2/2], Iter [10/14], reader_cost: 0.24073 s, batch_cost: 0.26355 s, ips: 15541.84990 samples/s, eta: 0:00:01
Epoch [2/2], Iter [11/14], reader_cost: 0.21886 s, batch_cost: 0.24141 s, ips: 16967.21446 samples/s, eta: 0:00:00
Epoch [2/2], Iter [12/14], reader_cost: 0.20063 s, batch_cost: 0.22291 s, ips: 18374.78776 samples/s, eta: 0:00:00
Epoch [2/2], Iter [13/14], reader_cost: 0.18521 s, batch_cost: 0.20728 s, ips: 19760.84536 samples/s, eta: 0:00:00
Epoch [2/2], Iter [14/14], reader_cost: 0.17199 s, batch_cost: 0.19436 s, ips: 21074.31905 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 3.68077 s, reader_cost: 2.40789 s, batch_cost: 2.72104 s, avg ips: 15579.36234 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.86450, Top5 accurary:: 0.99023
# 预期得到如下输出结果 - 推理输出
I0418 16:45:47.717545 85550 interpretercore.cc:267] New Executor is Running.
I0418 16:45:47.788849 85550 analysis_predictor.cc:1414] CustomDevice is enabled
--- Running analysis [ir_graph_build_pass]
I0418 16:45:47.790328 85550 executor.cc:186] Old Executor is Running.
--- Running analysis [ir_analysis_pass]
I0418 16:45:47.792423 85550 ir_analysis_pass.cc:53] argument has no fuse statis
--- Running analysis [ir_params_sync_among_devices_pass]
I0418 16:45:47.792572 85550 ir_params_sync_among_devices_pass.cc:142] Sync params from CPU to CustomDevicenpu/0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0418 16:45:47.880336 85550 analysis_predictor.cc:1565] ======= optimize end =======
I0418 16:45:47.880510 85550 naive_executor.cc:151] ---  skip [feed], feed -> inputs
I0418 16:45:47.881462 85550 naive_executor.cc:151] ---  skip [linear_5.tmp_1], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
```

## PaddleInference C++ 推理安装与运行

### PaddleInference C++ 预测库编译

> 注意：飞桨官网发布的 PaddleInference C++ 预测库中默认不含有 CustomDevice 功能支持，因此这里我们需要重新编译得到 PaddleInference C++ 预测库。

```bash
# 1) 进入 PaddlePaddle 主框架源码目录
cd PaddleCustomDevice/Paddle

# 2) 创建编译目录
mkdir build && cd build

# 3.1) X86-64 环境下的编译命令 - 编译 CPU 版本即可
cmake .. -DPY_VERSION=3.9 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_ARM=OFF
make -j8

# 3.2) Aarch64 环境下的编译命令 - 编译 CPU 版本即可
cmake .. -DPY_VERSION=3.9 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=ON \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_ARM=ON
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
| 调试     | CUSTOM_DEVICE_BLACK_LIST  | String   | 在黑名单内的算子会异构到CPU上运行 | "" |
| 调试     | FLAGS_npu_check_nan_inf | Bool   | 是否开启所有NPU算子输入输出检查   | False                                                        |
| 调试     | FLAGS_npu_blocking_run | Bool   | 是否开启强制同步执行所有 NPU 算子 | False                                                        |
| 性能分析 | FLAGS_npu_profiling_dir | String | 设置 Profiling 数据保存目录       | "ascend_profiling"                                           |
| 性能分析 | FLAGS_npu_profiling_dtypes | Uint64 | 指定需要采集的 Profiling 数据类型 | 见 [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L31) |
| 性能分析 | FLAGS_npu_profiling_metrics | Uint64 | 设置 AI Core 性能指标采集项       | 见 [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L36) |
| 性能加速 | FLAGS_npu_storage_format  | Bool   | 是否开启 Conv/BN 等算子的计算加速 | False                                                        |
