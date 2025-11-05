# 飞桨自定义接入硬件后端(天数GPU)

简体中文 | [English](./README.md)

请参考以下步骤进行编译安装与验证

## 编译安装

```bash
# 获请联系天数智芯客户支持(services@iluvatar.com)获取SDK镜像

# 克隆PaddleCustomDevice源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 设置环境变量
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib
export LIBRARY_PATH=/usr/local/corex/lib

# 编译 Paddle Custom Device
cd backends/iluvatar_gpu
bash build_paddle.sh

# 安装
bash install_paddle.sh
```

## 增量编译（代码修改后更快地重新编译）
```bash
# 增量编译（代码修改后更快地重新编译，会一并安装whl包）
bash build_inc.sh

# 清理构建环境（删除构建目录、还原补丁并重置状态）
bash build_inc.sh --clean
```

## 验证

```bash
# 运行测试
cd tests
bash run_test.sh
```
