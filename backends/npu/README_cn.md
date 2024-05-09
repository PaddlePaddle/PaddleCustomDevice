# 飞桨自定义接入硬件后端(昇腾NPU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(昇腾NPU)的编译安装与验证

## 昇腾NPU系统要求

| 芯片类型  | CANN版本     |
| --------- | -------- |
| 芯片类型 | 昇腾910 |
| CANN版本 | [CANN 7.0.1](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software) |
| 驱动版本 | [23.0.2](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software) |

**注意**：[release/2.6](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/release/2.6/backends/npu/README_cn.md) 分支仅支持『昇腾910』芯片，如需『昇腾910B』芯片的支持请切换到 [develop](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md) 分支进行编译安装。查看芯片类似请参考如下命令：

```bash
# 系统环境下运行如下命令，如果有设备列表输出，则表示当前为『昇腾910』芯片
lspci | grep d801

# 系统环境下运行如下命令，如果有设备列表输出，则表示当前为『昇腾910B』芯片
lspci | grep d802
```

## 环境准备与源码同步

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与 dockerfile 位于 tools/dockerfile 目录下
docker pull registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-x86_64-gcc84-py39
docker pull registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-aarch64-gcc84-py39

# 2) 参考如下命令启动容器
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash

# 3) 克隆 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice -b release/2.6
cd PaddleCustomDevice
```

## PaddlePaddle 安装与运行

### 源码编译安装

```bash
# 1) 进入硬件后端(昇腾NPU)目录
cd backends/npu

# 2) 编译之前需要先保证环境下装有飞桨安装包，直接安装飞桨 CPU 版本即可
# 默认开发镜像中不含有飞桨安装包，可通过如下地址安装 PaddlePaddle CPU 2.6.1 版本的安装包
pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 3) 编译选项，是否打开单元测试编译，默认值为 ON
export WITH_TESTING=OFF

# 4) 执行编译脚本 - submodule 在编译时会按需下载
bash tools/compile.sh

# 5) 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/paddle_custom_npu*.whl
```

### 基础功能检查

```bash
# 1. 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果
['npu']

# 2. 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# 预期得到如下输出结果
version: 2.6.1
commit: 79cd4ebe805a9a3c6bc7817a7ec2e1fee32ebe8e
cann: 7.0.1

# 3. 飞桨健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到如下输出结果
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

### 模型训练和推理示例

```bash
# 运行简单训练和推理示例
python tests/test_LeNet_MNIST.py

# 预期得到如下输出结果 - 训练输出
Epoch [1/2], Iter [01/14], reader_cost: 2.81928 s, batch_cost: 97.57224 s, ips: 41.97915 samples/s, eta: 0:45:32
Epoch [1/2], Iter [02/14], reader_cost: 1.41005 s, batch_cost: 48.79607 s, ips: 83.94119 samples/s, eta: 0:21:57
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.29687 s, batch_cost: 0.31025 s, ips: 13202.09133 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 4.88396 s, reader_cost: 4.15624 s, batch_cost: 4.34355 s, avg ips: 11741.29245 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.86462, Top5 accurary:: 0.99133

# 预期得到如下输出结果 - 推理输出
I0509 12:13:58.880553  9291 program_interpreter.cc:212] New Executor is Running.
I0509 12:13:58.911787  9291 analysis_predictor.cc:1658] CustomDevice is enabled
... ...
I0509 12:13:58.913389  9291 ir_params_sync_among_devices_pass.cc:144] Sync params from CPU to npu:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0509 12:13:58.917276  9291 analysis_predictor.cc:1838] ======= optimize end =======
I0509 12:13:58.917372  9291 naive_executor.cc:200] ---  skip [feed], feed -> inputs
I0509 12:13:58.917668  9291 naive_executor.cc:200] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
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
