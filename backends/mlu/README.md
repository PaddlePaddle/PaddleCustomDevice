# PaddlePaddle Custom Device Implementaion for Cambricon MLU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementaion for Cambricon MLU.

## Neuware Version

| Module    | Version  |
| --------- | -------- |
| cntoolkit | 3.8.4-1  |
| cnnl      | 1.23.2-1 |
| cnnlextra | 1.6.1-1  |
| cncl      | 1.14.0-1 |
| mluops    | 0.11.0-1 |

## Prepare environment and source code

```bash
# 1. pull PaddlePaddle Cambricon MLU development docker image
# dockerfile of the image is in tools/dockerfile directory
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.13.0-ubuntu20-gcc84-py310

# 2. refer to the following commands to start docker container
docker run -it --name paddle-mlu-dev -v `pwd`:/workspace \
    --shm-size=128G --network=host -w=/workspace \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --privileged -v /usr/bin/cnmon:/usr/bin/cnmon \
    registry.baidubce.com/device/paddle-mlu:ctr2.13.0-ubuntu20-gcc84-py310 /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle Installation and Verification

### Source Code Compilation

```bash
# 1. navigate to implementaion for Cambricon MLU
cd backends/mlu

# 2. before compiling, ensure that PaddlePaddle (CPU version) is installed, you can run the following command
pip install paddlepaddle==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script - submodules will be synced on demand when compile
bash tools/compile.sh

# 5. install the generated whl package, which is under build/dist directory
pip install build/dist/paddle_custom_mlu*.whl
```

## Verification

```bash
# 1. list available custom backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# output as following
['mlu']

# 2. check installed custom mlu version
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# output as following
version: 2.6.0
commit: 99fe60ff5e28788af73987027f7e3208fd5d7337
cntoolkit: 3.8.4
cnnl: 1.23.2
cnnlextra: 1.6.1
cncl: 1.14.0
mluops: 0.11.0

# 3. health check
python -c "import paddle; paddle.utils.run_check()"
# output as following
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## Train and Inference Demo

```bash
# demo for training, evaluation and inference
python tests/test_LeNet_MNIST.py

# training output as following
Epoch [1/2], Iter [01/14], reader_cost: 1.23499 s, batch_cost: 1.28666 s, ips: 3183.43983 samples/s, eta: 0:00:36
Epoch [1/2], Iter [02/14], reader_cost: 0.61760 s, batch_cost: 0.66744 s, ips: 6136.88837 samples/s, eta: 0:00:18
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.08951 s, batch_cost: 0.13775 s, ips: 29735.07966 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 1.96801 s, reader_cost: 1.25314 s, batch_cost: 1.92850 s, avg ips: 29138.02172 samples/s

# inference output as following
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

## Environment variables

| Name | Type | Desc | Default |
| ---- | ---- | ---- | ------- |
| PADDLE_MLU_ALLOW_TF32 | Bool | Whether to enable tf32 computation | True |
| CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE | Int | Whether to enlarge mem-pool of CNCL | 1 |
| CUSTOM_DEVICE_BLACK_LIST | String | op blacklist, force the operation to run in CPU mode | "" |
| FLAGS_allocator_strategy | ENUM | [paddlepaddle-doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/flags/memory_cn.html#flags-allocator-strategy) | auto_growth |
