# 飞桨自定义接入硬件后端(寒武纪MLU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(寒武纪MLU)的编译安装与验证

## Neuware 版本信息

| 模块名称  | 版本     |
| --------- | -------- |
| cntoolkit | 3.4.2-1  |
| cnnl      | 1.17.0-1 |
| cncl      | 1.9.3-1  |
| mluops    | 0.6.0-1  |

## 环境准备与源码同步

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与 dockerfile 位于 tools/dockerfile 目录下
docker pull registry.baidubce.com/device/paddle-mlu:cntoolkit3.4.2-1-cnnl1.17.0-1-gcc82

# 2) 参考如下命令启动容器
docker run -it --name paddle-mlu-dev -v `pwd`:/workspace \
    --shm-size=128G --network=host -w=/workspace \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --privileged -v /usr/bin/cnmon:/usr/bin/cnmon \
    registry.baidubce.com/device/paddle-mlu:cntoolkit3.4.2-1-cnnl1.17.0-1-gcc82 /bin/bash


# 3) 克隆 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle 安装与运行

> 注意：此步骤编译得到的 PaddlePaddle Python WHL 安装包同时包含训练和推理功能，其中推理仅支持 PaddleInference Python API，如果需要 PaddleInference C++ API 请参考下一章节 "PaddleInference C++ 推理安装与运行"。

### 编译安装

```bash
# 1) 进入硬件后端(寒武纪MLU)目录
cd backends/mlu

# 2) 编译之前需要先保证环境下装有飞桨安装包，直接安装飞桨 CPU 版本即可
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 3) 编译选项，是否打开单元测试编译，默认值为 ON
export WITH_TESTING=OFF

# 4) 执行编译脚本 - submodule 在编译时会按需下载
bash tools/compile.sh

# 5) 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/paddle_custom_mlu*.whl
```

### 功能验证

```bash
# 1) 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果
['mlu']

# 2) 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: 98ae7a84b51e36fc15a1fef7808a82ed8b792fdf
cntoolkit: 3.4.2
cnnl: 1.17.0
cncl: 1.9.3
mluops: 0.6.0

# 3) 运行简单模型训练、评估和推理任务
python tests/test_LeNet_MNIST.py
# 预期得到如下输出结果 - 训练输出
Epoch [1/2], Iter [01/14], reader_cost: 1.23499 s, batch_cost: 1.28666 s, ips: 3183.43983 samples/s, eta: 0:00:36
Epoch [1/2], Iter [02/14], reader_cost: 0.61760 s, batch_cost: 0.66744 s, ips: 6136.88837 samples/s, eta: 0:00:18
... ...
Epoch [2/2], Iter [10/14], reader_cost: 0.12527 s, batch_cost: 0.17382 s, ips: 23565.25098 samples/s, eta: 0:00:00
Epoch [2/2], Iter [11/14], reader_cost: 0.11389 s, batch_cost: 0.16232 s, ips: 25233.58550 samples/s, eta: 0:00:00
Epoch [2/2], Iter [12/14], reader_cost: 0.10441 s, batch_cost: 0.15278 s, ips: 26810.42007 samples/s, eta: 0:00:00
Epoch [2/2], Iter [13/14], reader_cost: 0.09639 s, batch_cost: 0.14467 s, ips: 28312.30064 samples/s, eta: 0:00:00
Epoch [2/2], Iter [14/14], reader_cost: 0.08951 s, batch_cost: 0.13775 s, ips: 29735.07966 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 1.96801 s, reader_cost: 1.25314 s, batch_cost: 1.92850 s, avg ips: 29138.02172 samples/s
# 预期得到如下输出结果 - 推理输出
I0705 17:13:54.706485 42228 program_interpreter.cc:171] New Executor is Running.
I0705 17:13:54.730235 42228 analysis_predictor.cc:1502] CustomDevice is enabled
--- Running analysis [ir_graph_build_pass]
I0705 17:13:54.730543 42228 executor.cc:187] Old Executor is Running.
--- Running analysis [ir_analysis_pass]
I0705 17:13:54.730988 42228 ir_analysis_pass.cc:46] argument has no fuse statis
--- Running analysis [save_optimized_model_pass]
W0705 17:13:54.731001 42228 save_optimized_model_pass.cc:28] save_optim_cache_model is turned off, skip save_optimized_model_pass
--- Running analysis [ir_params_sync_among_devices_pass]
I0705 17:13:54.731010 42228 ir_params_sync_among_devices_pass.cc:142] Sync params from CPU to mlu:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0705 17:13:54.733326 42228 analysis_predictor.cc:1676] ======= optimize end =======
I0705 17:13:54.733381 42228 naive_executor.cc:171] ---  skip [feed], feed -> inputs
I0705 17:13:54.733578 42228 naive_executor.cc:171] ---  skip [linear_5.tmp_1], fetch -> fetch
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
# 1) 进入硬件后端(寒武纪MLU)目录
cd backends/mlu

# 2) 编译选项，PADDLE_INFERENCE_LIB_DIR 为上一步编译得到的 C++ 预测库的地址
export ON_INFER=ON # 是否打开推理库编译，默认为 OFF
export PADDLE_INFERENCE_LIB_DIR=/path/to/Paddle/build/paddle_inference_install_dir

# 3) 执行编译脚本
bash tools/compile.sh

# 4) 编译产出为 build 目录下的 libpaddle-custom-mlu.so 文件，指定插件路径到库文件目录下
export CUSTOM_DEVICE_ROOT=/path/to/PaddleCustomDevice/backends/mlu/build
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

# 4) 修改 resnet50_test.cc，使用 mlu 作为设备传入
config.EnableCustomDevice("mlu", 0);

# 5) 修改 compile.sh 编译文件，需根据 C++ 预测库的 version.txt 信息对以下的几处内容进行修改
WITH_MKL=ON  # 如果是 Aarch 环境，请设置为 OFF
WITH_GPU=OFF
WITH_ARM=OFF # 如果是 Aarch 环境，请设置为 ON

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

## 环境变量

### PADDLE_MLU_ALLOW_TF32
该功能使Conv，MatMul类算子以TF32数据类型进行计算，目前只支持MLU590板卡，TF32是MLU590运行的默认数据类型。

开启TF32数据类型计算。
```bash
export PADDLE_MLU_ALLOW_TF32=true
```

### CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE
该功能会关闭CNCL对于多通信域（CLIQUE）管理的内存限制，CNCL为了节省内存会默认限制进程为单通信域，在某些需要管理多通信域的集合通信中会出现CNCL初始化卡死。

关闭多通信域内存限制。
```bash
export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1
```
