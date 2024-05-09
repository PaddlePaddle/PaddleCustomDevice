# PaddlePaddle Custom Device Implementation for Ascend NPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Ascend NPU.

## Ascend NPU System Requirements

| Type | Version     |
| --------- | -------- |
| Chip | Ascend 910 |
| CANN | [CANN 7.0.1](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software) |
| Driver | [23.0.2](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software) |

**Note**：[release/2.6](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/release/2.6/backends/npu/README_cn.md) branch only supports 'Ascend910' chip, please swith to [develop](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md) branch if you want to use 'Ascend910B' chip. Please refer to the flollowing commands to check chip version:

```bash
# check if the chip version is 'Ascend910'
lspci | grep d801

# check if the chip version is 'Ascend910B'
lspci | grep d802
```

## Prepare environment and source code

```bash
# 1. pull PaddlePaddle Ascend NPU development docker image
# dockerfile of the image is in tools/dockerfile directory
docker pull registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-x86_64-gcc84-py39
docker pull registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-aarch64-gcc84-py39

# 2. refer to the following commands to start docker container
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice -b release/2.6
cd PaddleCustomDevice
```

## PaddlePaddle Installation and Verification

### Source Code Compile

```bash
# 1. go to ascend npu directory
cd backends/npu

# 2. please ensure the PaddlePaddle cpu whl package is already installed
# the development docker image NOT have PaddlePaddle cpu whl installed by default
# you may download and install the PaddlePaddle CPU 2.6.1 whl package with links below
pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script - submodules will be synced on demand when compile
bash tools/compile.sh

# 5. install the generated whl package, which is under build/dist directory
pip install build/dist/paddle_custom_npu*.whl
```

## Verification

```bash
# 1. list available custom backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# output as following
['npu']

# 2. check installed custom npu version
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# output as following
version: 2.6.1
commit: 79cd4ebe805a9a3c6bc7817a7ec2e1fee32ebe8e
cann: 7.0.1

# 3. health check
python -c "import paddle; paddle.utils.run_check()"
# output as following
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## Train and Inference Demo

```bash
# demo for training, evaluation and inference
python tests/test_LeNet_MNIST.py

# training output as following
Epoch [1/2], Iter [01/14], reader_cost: 2.81928 s, batch_cost: 97.57224 s, ips: 41.97915 samples/s, eta: 0:45:32
Epoch [1/2], Iter [02/14], reader_cost: 1.41005 s, batch_cost: 48.79607 s, ips: 83.94119 samples/s, eta: 0:21:57
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.29687 s, batch_cost: 0.31025 s, ips: 13202.09133 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 4.88396 s, reader_cost: 4.15624 s, batch_cost: 4.34355 s, avg ips: 11741.29245 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.86462, Top5 accurary:: 0.99133

# inference output as following
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

## Environment Variables

| Subject     | Variable Name       | Type   | Description    | Default Value |
| -------- | -------------------------------- | ------ | --------------------------------- | ------------------------------------------------------------ |
| Debug     | CUSTOM_DEVICE_BLACK_LIST| String | Ops in back list will fallbacks to CPU  |  ""  |
| Debug     | FLAGS_npu_check_nan_inf | Bool   | check nan or inf of all npu kernels | False                                                       |
| Debug     | FLAGS_npu_blocking_run | Bool   | enable sync for all npu kernels | False                                                     |
| Profiling | FLAGS_npu_profiling_dir | String |   ACL profiling output dir     | "ascend_profiling"                                           |
| Profiling | FLAGS_npu_profiling_dtypes | Uint64 | ACL datatypes to profile | Refer to [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L31) |
| Profiling | FLAGS_npu_profiling_metrics | Uint64 | AI Core metric to profile  | Refer to [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L36) |
| Performance | FLAGS_npu_storage_format         | Bool   | enable Conv/BN acceleration | False                                                        |
