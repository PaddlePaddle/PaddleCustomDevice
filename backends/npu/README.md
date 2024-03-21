# PaddlePaddle Custom Device Implementation for Ascend NPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Ascend NPU.

## Ascend NPU System Requirements

| Type | Version     |
| --------- | -------- |
| Chip | Ascend 910A、Ascend 910B |
| CANN | [CANN 7.0.1](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261956975) |
| Driver | [23.0.1](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/261964443) |
| Firmware | [7.1.0.4.220](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/261964443) |

## Prepare environment and source code

```bash
# 1. pull PaddlePaddle Ascend NPU development docker image
# dockerfile of the image is in tools/dockerfile directory
# Ascend 910A - check with the output of 'lspci | grep d801'
registry.baidubce.com/device/paddle-npu:cann701-910A-ubuntu18-x86_64
registry.baidubce.com/device/paddle-npu:cann701-910A-ubuntu18-aarch64
# Ascend 910B - check with the output of 'lspci | grep d802'
registry.baidubce.com/device/paddle-npu:cann701-910B-ubuntu18-x86_64
registry.baidubce.com/device/paddle-npu:cann701-910B-ubuntu18-aarch64

# 2. refer to the following commands to start docker container
docker run -it --name paddle-dev -v `pwd`:/work -w=/work \
    --privileged --network=host --shm-size=128G \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann700-910B-ubuntu18-$(uname -m) /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle Installation and Verification

### Source Code Compile

```bash
# 1. go to ascend npu directory
cd backends/npu

# 2. please ensure the PaddlePaddle cpu whl package is already installed
# the development docker image NOT have PaddlePaddle cpu whl installed by default
# you may download and install the nightly built cpu whl package with links below
https://paddle-device.bj.bcebos.com/0.0.0/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl
https://paddle-device.bj.bcebos.com/0.0.0/cpu/paddlepaddle-0.0.0-cp39-cp39-linux_aarch64.whl

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
version: 0.0.0
commit: 75ee24202649770d11860b52d9b06366b9358a2f
cann: 7.0.0

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
Epoch [1/2], Iter [01/14], reader_cost: 1.00473 s, batch_cost: 74.52487 s, ips: 54.96152 samples/s, eta: 0:34:46
Epoch [1/2], Iter [02/14], reader_cost: 0.50253 s, batch_cost: 37.26772 s, ips: 109.90745 samples/s, eta: 0:16:46
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.10162 s, batch_cost: 0.10956 s, ips: 37384.97312 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 1.79473 s, reader_cost: 1.42272 s, batch_cost: 1.53388 s, avg ips: 31951.29112 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.85315, Top5 accurary:: 0.98914

# inference output as following
I0103 19:11:34.570551 43520 program_interpreter.cc:214] New Executor is Running.
I0103 19:11:34.589533 43520 analysis_predictor.cc:1684] CustomDevice is enabled
... ...
I0103 19:11:34.590348 43520 ir_params_sync_among_devices_pass.cc:144] Sync params from CPU to npu:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0103 19:11:34.592526 43520 analysis_predictor.cc:1867] ======= optimize end =======
I0103 19:11:34.592568 43520 naive_executor.cc:200] ---  skip [feed], feed -> inputs
I0103 19:11:34.592684 43520 naive_executor.cc:200] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
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
| Performance | FLAGS_npu_storage_format         | Bool   | enable Conv/BN private ACL format | False                                                        |
| OP Compile | FLAGS_npu_jit_compile  | Bool   | enable NPU OP JIT compile  | True |
