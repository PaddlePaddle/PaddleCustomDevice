# 飞桨自定义接入硬件后端(寒武纪MLU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(寒武纪MLU)的编译安装与验证

## Neuware 版本信息

| 模块名称  | 版本     |
| --------- | -------- |
| cntoolkit | 3.8.4-1  |
| cnnl      | 1.23.2-1 |
| cnnlextra | 1.6.1-1  |
| cncl      | 1.14.0-1 |
| mluops    | 0.11.0-1 |

## 环境准备与源码同步

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与 dockerfile 位于 tools/dockerfile 目录下
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.13.0-ubuntu20-gcc84-py310

# 2) 参考如下命令启动容器
docker run -it --name paddle-mlu-dev -v `pwd`:/workspace \
    --shm-size=128G --network=host -w=/workspace \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --privileged -v /usr/bin/cnmon:/usr/bin/cnmon \
    registry.baidubce.com/device/paddle-mlu:ctr2.13.0-ubuntu20-gcc84-py310 /bin/bash

# 3) 克隆 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle 安装与运行

### 源码编译安装

```bash
# 1) 进入硬件后端(寒武纪MLU)目录
cd backends/mlu

# 2) 编译之前需要先保证环境下装有飞桨安装包，直接安装飞桨 CPU 版本即可
pip install paddlepaddle==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 编译选项，是否打开单元测试编译，默认值为 ON
export WITH_TESTING=OFF

# 4) 执行编译脚本 - submodule 在编译时会按需下载
bash tools/compile.sh

# 5) 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/paddle_custom_mlu*.whl
```

### 基础功能检查

```bash
# 1) 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果
['mlu']

# 2) 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# 预期得到如下输出结果
version: 2.6.0
commit: 55f88ebff9297f2f4b90d61e211d2cf2784f2ad9
cntoolkit: 3.8.4
cnnl: 1.23.2
cnnlextra: 1.6.1
cncl: 1.14.0
mluops: 0.11.0
# 3. 飞桨健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到如下输出结果
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

### 模型训练和推理示例

```bash
# 运行简单训练和推理示例
python tests/test_LeNet_MNIST.py

# 预期得到如下输出结果 - 训练输出
Epoch [1/2], Iter [01/14], reader_cost: 1.23499 s, batch_cost: 1.28666 s, ips: 3183.43983 samples/s, eta: 0:00:36
Epoch [1/2], Iter [02/14], reader_cost: 0.61760 s, batch_cost: 0.66744 s, ips: 6136.88837 samples/s, eta: 0:00:18
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.08951 s, batch_cost: 0.13775 s, ips: 29735.07966 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 1.96801 s, reader_cost: 1.25314 s, batch_cost: 1.92850 s, avg ips: 29138.02172 samples/s

# 预期得到如下输出结果 - 推理输出
I0705 17:13:54.706485 42228 program_interpreter.cc:171] New Executor is Running.
I0705 17:13:54.730235 42228 analysis_predictor.cc:1502] CustomDevice is enabled
... ...
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

## 环境变量

| 名称 | 类型 | 描述 | 默认值 |
| ---- | ---- | ---- | ------- |
| PADDLE_MLU_ALLOW_TF32 | Bool | 是否开启tf32计算 | True |
| CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE | Int | 是否增大CNCL的内存池 | 1 |
| CUSTOM_DEVICE_BLACK_LIST | String | 算子黑名单，在黑名单上的算子会强制在CPU上执行 | "" |
| FLAGS_allocator_strategy | ENUM | [飞桨官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/flags/memory_cn.html#flags-allocator-strategy) | auto_growth |
