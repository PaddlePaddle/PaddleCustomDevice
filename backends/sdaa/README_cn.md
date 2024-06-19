# 飞桨自定义接入硬件后端(太初SDAA)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(太初SDAA)的编译安装与验证

## 太初SDAA系统要求

| 组件         | 版本     |
| ---------   | -------- |
| TecoDriver  | 1.1.0    |
| TecoToolkit | 1.1.0    |

## 环境准备与源码同步
```bash
# 1. 拉取开发镜像
wget http://mirrors.tecorigin.com/repository/teco-docker-tar-repo/release/ubuntu22.04/x86_64/1.1.0/paddle-1.1.0-paddle_sdaa1.1.0.tar
docker load < paddle-1.1.0-paddle_sdaa1.1.0.tar

# 2. 参考如下命令启动容器并激活 conda 环境（conda 环境中已安装 PaddlePaddle 主框架）
docker run -it --name="paddle_sdaa_dev" --net=host -v $(pwd):/work \
--device=/dev/tcaicard0 --device=/dev/tcaicard1 \
--device=/dev/tcaicard2 --device=/dev/tcaicard3 \
--cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 64g \
jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/paddle:1.1.0-paddle_sdaa1.1.0 /bin/bash

conda activate paddle_env

# 3. 克隆 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## 安装与运行

### 源码编译安装

```bash
# 1. 更新 cmake
pip install -U cmake

# 2. 切换到 develop 分支
git checkout develop

# 3. 执行以下指令更新子模块代码
git submodule sync
git submodule update --init --recursive

# 4. 进入硬件后端(太初SDAA)目录
cd backends/sdaa

# 5. 执行编译脚本
bash compile.sh

# 6. 编译产出在 build/dist 路径下，使用 pip 安装
pip install -U build/dist/*.whl
```

### 基础功能检查

```bash
# 1. 列出可用硬件后端
python3 -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果:
['sdaa']

# 2. 使用 paddle_sdaa utils 模块的 `run_check` 功能检查 paddle-sdaa 插件和 PaddlePaddle主框架是否正常安装
python3 -c "import paddle_sdaa; paddle_sdaa.utils.run_check()"
# 预期得到如下输出结果:
paddle-sdaa and paddlepaddle are installed successfully!

# 3. 运行 relu 前向计算
python3 -c "import paddle;paddle.set_device('sdaa');print(paddle.nn.functional.relu(paddle.to_tensor([-2., 1.])))"
# 预期得到如下输出结果:
Tensor(shape=[2], dtype=float32, place=Place(sdaa:0), stop_gradient=True,
       [0., 1.])
```

## 模型训练和推理示例

```bash
# 运行简单训练和推理示例
python tests/test_MNIST_model.py

# 预期得到如下输出结果 - 训练输出
Epoch [1/2], Iter [01/14], reader_cost: 1.41201 s, batch_cost: 1.56096 s, ips: 2624.03256 samples/s, eta: 0:00:43
Epoch [1/2], Iter [02/14], reader_cost: 0.70611 s, batch_cost: 0.84809 s, ips: 4829.67512 samples/s, eta: 0:00:22
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.11122 s, batch_cost: 0.24438 s, ips: 16760.81762 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 3.50429 s, reader_cost: 1.55708 s, batch_cost: 3.42131 s, avg ips: 16363.92196 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.84607, Top5 accurary:: 0.98462

# 预期得到如下输出结果 - 推理输出
I0307 05:21:33.673595  6583 interpretercore.cc:237] New Executor is Running.
I0307 05:21:33.703184  6583 analysis_predictor.cc:1503] CustomDevice is enabled
... ...
I0307 05:21:33.707281  6583 analysis_predictor.cc:1660] ======= optimize end =======
I0307 05:21:33.707347  6583 naive_executor.cc:164] ---  skip [feed], feed -> inputs
I0307 05:21:33.707659  6583 naive_executor.cc:164] ---  skip [linear_5.tmp_1], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
```

## 环境变量

| 主题     | 变量名称       | 类型   | 描述    | 默认值 |
| -------- | -------------------------------- | ------ | --------------------------------- | ------------------------------------------------------------ |
| 调试     | CUSTOM_DEVICE_BLACK_LIST| String | 在黑名单内的算子会异构到CPU上运行  |  ""  |
| 性能分析     | ENABLE_SDPTI | String | 设置是否通过SDPTI获取设备端时间 | 1 |
| 调试     | HIGH_PERFORMANCE_CONV | String | 设置是否开启高性能卷积计算 | 0 |
| 调试     | FLAGS_sdaa_runtime_debug    | bool   | 打印运行时debug信息 | false |
| 功能   | FLAGS_sdaa_reuse_event      | bool   | 设置是否使用Event Pool功能         | true  |
