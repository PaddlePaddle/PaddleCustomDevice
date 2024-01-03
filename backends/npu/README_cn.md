# 飞桨自定义接入硬件后端(昇腾NPU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(昇腾NPU)的编译安装与验证

## 昇腾NPU系统要求

| 芯片类型  | CANN版本     |
| --------- | -------- |
| 昇腾910A | [CANN 7.0.RC1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.RC1.beta1)  |
| 昇腾910B | [CANN 7.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=7.0.0)  |

## 环境准备与源码同步

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与 dockerfile 位于 tools/dockerfile 目录下
# 昇腾910A芯片请使用如下镜像
registry.baidubce.com/device/paddle-npu:cann70RC1-910A-ubuntu18-x86_64
registry.baidubce.com/device/paddle-npu:cann70RC1-910A-ubuntu18-aarch64
# 昇腾910B芯片请使用如下镜像
registry.baidubce.com/device/paddle-npu:cann700-910B-ubuntu18-x86_64
registry.baidubce.com/device/paddle-npu:cann700-910B-ubuntu18-aarch64

# 2) 参考如下命令启动容器
docker run -it --name paddle-dev -v `pwd`:/workspace -w=/workspace \
    --privileged --network=host --shm-size=128G \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    registry.baidubce.com/device/paddle-npu:cann700-910B-ubuntu18-$(uname -m) /bin/bash

# 3) 克隆 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle 安装与运行

### 源码编译安装

```bash
# 1) 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 2) 编译之前需要先保证环境下装有飞桨安装包，直接安装飞桨 CPU 版本即可
# 默认开发镜像中不含有飞桨安装包，可通过如下地址安装 PaddlePaddle develop 分支的 nightly build 版本的安装包
https://paddle-device.bj.bcebos.com/0.0.0/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl
https://paddle-device.bj.bcebos.com/0.0.0/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_aarch64.whl

# 3) 编译选项，是否打开单元测试编译，默认值为 ON
export WITH_TESTING=OFF

# 4) 执行编译脚本 - submodule 在编译时会按需下载
bash tools/compile.sh

# 5) 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/paddle_custom_npu*.whl
```

### 基础功能检查

```bash
# 1) 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果
['npu']

# 2) 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: dbfe1b6fc559abc6f20ab0bf14e93e7fcdca7001
cann: 7.0.RC1

# 3) 飞桨健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到如下输出结果
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

# 4) 运行简单训练和推理示例
python tests/test_LeNet_MNIST.py
# 预期得到如下输出结果 - 训练输出
Epoch [1/2], Iter [01/14], reader_cost: 2.27062 s, batch_cost: 14.45539 s, ips: 283.35449 samples/s, eta: 0:06:44
Epoch [1/2], Iter [02/14], reader_cost: 1.13547 s, batch_cost: 7.23942 s, ips: 565.79091 samples/s, eta: 0:03:15
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.17199 s, batch_cost: 0.19436 s, ips: 21074.31905 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 3.68077 s, reader_cost: 2.40789 s, batch_cost: 2.72104 s, avg ips: 15579.36234 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.86450, Top5 accurary:: 0.99023
# 预期得到如下输出结果 - 推理输出
I0418 16:45:47.717545 85550 interpretercore.cc:267] New Executor is Running.
I0418 16:45:47.788849 85550 analysis_predictor.cc:1414] CustomDevice is enabled
... ...
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

### 模型训练示例

```bash
# 下载 PaddleClas 模型代码并安装Python依赖库
git clone https://github.com/PaddlePaddle/PaddleClas.git && cd PaddleClas
pip install --upgrade -r requirements.txt

# 下载训练数据集，并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip
unzip flowers102.zip

# 运行模型训练
python tools/train.py -c ppcls/configs/quick_start/ResNet50_vd.yaml \
    -o Arch.pretrained=True \
    -o Global.device=npu

# 得到训练精度结果如下
# The best top1 acc 0.9402, in epoch: 20
```

### 模型推理示例

```bash
# 在上一步训练完成之后，使用训练出来的模型进行推理
cd PaddleClas
python tools/infer.py -c ppcls/configs/quick_start/ResNet50_vd.yaml \
       -o Infer.infer_imgs=dataset/flowers102/jpg/image_00001.jpg \
       -o Global.pretrained_model="output/ResNet50_vd/best_model" \
       -o Global.device=npu

# 得到推理结果如下
# [{'class_ids': [76, 9, 11, 12, 13], 'scores': [0.98354, 0.00209, 0.00194, 0.00146, 0.00124], 'file_name': 'dataset/flowers102/jpg/image_00001.jpg', 'label_names': ['passion flower', 'globe thistle', "colt's foot", 'king protea', 'spear thistle']}]
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
| 性能加速 | FLAGS_npu_storage_format  | Bool   | 支持 Conv/BN 等算子的昇腾私有化格式 | False                                                        |
