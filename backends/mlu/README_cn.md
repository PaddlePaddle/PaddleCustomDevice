# 飞桨自定义接入硬件后端(寒武纪MLU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(寒武纪MLU)的编译安装与验证

## Neuware 版本信息

| 模块名称  | 版本     |
| --------- | -------- |
| cntoolkit | 3.10.2-1  |
| cnnl      | 1.25.1-1 |
| cnnlextra | 1.8.1-1  |
| cncl      | 1.16.0-1 |
| mluops    | 1.1.1-1 |

## 环境准备与源码同步

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与 dockerfile 位于 tools/dockerfile 目录下
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310

# 2) 参考如下命令启动容器
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310 /bin/bash

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
# 默认开发镜像中不含有飞桨安装包，可通过如下地址安装 PaddlePaddle develop 分支的 nightly build 版本的安装包
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

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
version: 0.0.0
commit: 83dfe3de33f0a915fb189161568fc3804b5f9c1b
cntoolkit: 3.10.2
cnnl: 1.25.1
cnnlextra: 1.8.1
cncl: 1.16.0
mluops: 1.1.1

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
Epoch [1/2], Iter [01/14], reader_cost: 2.73611 s, batch_cost: 2.78069 s, ips: 1473.01483 samples/s, eta: 0:01:17
Epoch [1/2], Iter [02/14], reader_cost: 1.37505 s, batch_cost: 1.41733 s, ips: 2889.94454 samples/s, eta: 0:00:38
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.19809 s, batch_cost: 0.23765 s, ips: 17235.35966 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 3.46918 s, reader_cost: 2.77321 s, batch_cost: 3.32711 s, avg ips: 16529.56425 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.86230, Top5 accurary:: 0.98950

# 预期得到如下输出结果 - 推理输出
I0521 20:27:26.487897  2030 program_interpreter.cc:221] New Executor is Running.
I0521 20:27:26.499172  2030 analysis_predictor.cc:1850] CustomDevice is enabled
... ...
I0521 20:27:26.500521  2030 ir_params_sync_among_devices_pass.cc:142] Sync params from CPU to mlu:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [save_optimized_model_pass]
--- Running analysis [ir_graph_to_program_pass]
I0521 20:27:26.504653  2030 analysis_predictor.cc:2032] ======= ir optimization completed =======
I0521 20:27:26.504727  2030 naive_executor.cc:200] ---  skip [feed], feed -> inputs
I0521 20:27:26.504953  2030 naive_executor.cc:200] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
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
