# PaddlePaddle Custom Device Implementation for Tecorigin SDAA

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Tecorigin SDAA.

## Tecorigin SDAA System Requirements

| Module      | Version  |
| ---------   | -------- |
| TecoDriver  | 1.1.0    |
| TecoToolkit | 1.1.0    |

## Prepare environment and source code
```bash
# 1. pull PaddlePaddle Tecorigin SDAA development docker image
wget http://mirrors.tecorigin.com/repository/teco-docker-tar-repo/release/ubuntu22.04/x86_64/1.1.0/paddle-1.1.0-paddle_sdaa1.1.0.tar
docker load < paddle-1.1.0-paddle_sdaa1.1.0.tar

# 2. refer to the following commands to start docker container and activate conda environment (PaddlePaddle framwork has been installed in the conda environment)
docker run -it --name="paddle_sdaa_dev" --net=host -v $(pwd):/work \
--device=/dev/tcaicard0 --device=/dev/tcaicard1 \
--device=/dev/tcaicard2 --device=/dev/tcaicard3 \
--cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 64g \
jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/paddle:1.1.0-paddle_sdaa1.1.0 /bin/bash

conda activate paddle_env

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## Installation and Verification

### Source Code Compile

```bash
# 1. update cmake
pip install -U cmake

# 2. checkout branch to `develop`
git checkout develop

# 3. execute the following commands to update submodule
git submodule sync
git submodule update --init --recursive

# 4. go to Tecorigin sdaa directory
cd backends/sdaa

# 5. execute compile script
bash compile.sh

# 6. install the generated whl package, which is under build/dist directory
pip install build/dist/*.whl --force-reinstall
```

### Verification

```bash
# 1. using paddle_sdaa utils's `run_check` to check whether paddle-sdaa plugin and PaddlePaddle framework are installed.
python3 -c "import paddle_sdaa; paddle_sdaa.utils.run_check()"
# expected output:
paddle-sdaa and paddlepaddle are installed successfully!


# 2. list available hardware backends
python3 -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# expected ouput:
['sdaa']

# 3. run relu forward
python3 -c "import paddle;paddle.set_device('sdaa');print(paddle.nn.functional.relu(paddle.to_tensor([-2., 1.])))"
# expected output:
Tensor(shape=[2], dtype=float32, place=Place(sdaa:0), stop_gradient=True,
       [0., 1.])
```

## Train and Inference Demo

```bash
# demo for training, evaluation and inference
python tests/test_MNIST_model.py

# training output as following
Epoch [1/2], Iter [01/14], reader_cost: 1.41201 s, batch_cost: 1.56096 s, ips: 2624.03256 samples/s, eta: 0:00:43
Epoch [1/2], Iter [02/14], reader_cost: 0.70611 s, batch_cost: 0.84809 s, ips: 4829.67512 samples/s, eta: 0:00:22
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.11122 s, batch_cost: 0.24438 s, ips: 16760.81762 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 3.50429 s, reader_cost: 1.55708 s, batch_cost: 3.42131 s, avg ips: 16363.92196 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.84607, Top5 accurary:: 0.98462

# inference output as following
I0307 05:21:33.673595  6583 interpretercore.cc:237] New Executor is Running.
I0307 05:21:33.703184  6583 analysis_predictor.cc:1503] CustomDevice is enabled
... ...
I0307 05:21:33.707281  6583 analysis_predictor.cc:1660] ======= optimize end =======
I0307 05:21:33.707347  6583 naive_executor.cc:164] ---  skip [feed], feed -> inputs
I0307 05:21:33.707659  6583 naive_executor.cc:164] ---  skip [linear_5.tmp_1], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
```

## Environment Variables

| Subject     | Variable Name       | Type   | Description    | Default Value |
| -------- | -------------------------------- | ------ | --------------------------------- | ------------------------------------------------------------ |
| Debug     | CUSTOM_DEVICE_BLACK_LIST| String | Ops in back list will fallbacks to CPU  |  ""  |
| Profiling     | ENABLE_SDPTI | String | enable sdpti | 1 |
| Debug     | HIGH_PERFORMANCE_CONV | String | set HIGH_PERFORMANCE_CONV to `"1"` can enable high performance conv API | 0 |
| Debug     | FLAGS_sdaa_runtime_debug    | bool   | print runtime information | false |
| Feature   | FLAGS_sdaa_reuse_event      | bool   | enable event pool         | true  |
