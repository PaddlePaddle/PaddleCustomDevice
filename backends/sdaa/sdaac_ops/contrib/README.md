# 自定义 SDAA C 算子贡献指南

SDAA C 算子是使用 SDAA C 编程模型实现的能够在 Teco 异构并行计算平台上完成特定计算任务的核函数。本指南通过示例的形式介绍在硬件后端(太初 SDAA)中接入自定义 SDAA C 算子的全流程，参照本指南，您可以通过自定义 SDAA C 算子在太初 SDAA 硬件上实现您需要的计算逻辑。我们非常欢迎您为我们贡献您的宝贵代码！

## 环境准备

- tecocc >= 1.9.0

## 自定义 SDAA C 算子示例

下面我们通过实现一个三角函数 **cos** 的例子来介绍如何将自定义 SDAA C 算子接入太初 SDAA。

### 实现完成计算功能的 SDAA Kernel

首先，我们需要在 `backends/sdaa/sdaac_ops/contrib` 目录下创建以 `.scpp` 结尾的 SDAA C 文件并实现其功能。比如我们创建实现三角函数 **cos** 计算功能的 SDAA Kernel `custom_cos.scpp`，其代码如下(本指南仅介绍 SDAA C 算子接入太初硬件，SADD C 编程可以参考[官方文档](http://docs.tecorigin.com/release/sdaac/v1.10.0/))：


```cpp
#include "sdaacops_contrib.h"

__global__ void MyCos(const float *x, float *y, int num) {
     sdaa::sync_threads();

    if (threadDim >= num) {
        y[threadIdx] = cosf(x[threadIdx]);
    } else {
        int cal_times = num / threadDim;
        for (size_t i = 0; i < cal_times; ++i) {
            y[i * threadDim + threadIdx] = cosf(x[i * threadDim + threadIdx]);
        }
        int remain_times = num % threadDim;
        if (threadIdx < remain_times) {
            y[threadDim * cal_times + threadIdx] = cosf(x[threadDim * cal_times + threadIdx]);
        }
    }

    sdaa::sync_threads();
}

void custom_sdaa_cos_forward(sdaaStream_t stream,
                             const float *x,
                             float *y,
                             int num) {
  MyCos<<<1, stream>>>(x, y, num);
  return;
}
```

其中 `MyCos` 是 device 侧实现计算功能的函数，`custom_sdaa_cos_forward` 是 host 端的调用函数。实现 `custom_cos.scpp` 后，还需将调用函数 `custom_sdaa_cos_forward` 的声明写入到同路径下的头文件 `sdaacops_contrib.h` 中：

```cpp
#pragma once
#include <sdaa_runtime.h>

void custom_sdaa_cos_forward(sdaaStream_t stream,
                             const float *x,
                             float *y,
                             int num);
```

### 实现 PaddlePaddle 自定义 C++ 算子

自定义 SDAA C 算子实现之后，还需要进行调用。这里通过 PaddlePaddle 的自定义 C++ 算子功能实现定制化算子的接入，因此还需要编写 PaddlePaddle 自定义 C++ 算子来调用前面实现的 SDAA C 算子。PaddlePaddle 自定义 C++ 算子的写法可以参考[官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html)，这里不详细展开。我们在 `backends/sdaa/sdaa_ext` 目录下新建源文件 `sdaa_custom_cos.cc`，代码如下：

```cpp
#include <vector>

#include "custom_sdaacops.h"  // NOLINT
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"
#include "runtime/runtime.h"

#define CHECK_CUSTOM_INPUT(x) \
  PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

std::vector<paddle::Tensor> cos_SDAA_forward(const paddle::Tensor& x) {
  CHECK_CUSTOM_INPUT(x);
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  sdaaStream_t custom_stream =
      static_cast<CustomSDAAStream_t>(dev_ctx->stream())->pStream;
  auto out = paddle::empty_like(x, x.dtype(), x.place());

  custom_sdaa_cos_forward(
      custom_stream, x.data<float>(), out.data<float>(), x.numel());

  return {paddle::Tensor(out)};
}

PD_BUILD_OP(custom_cos)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(cos_SDAA_forward));
```

其中 `PD_BUILD_OP` 宏后面的括号内为算子名，也是后面在 python 端调用的接口名。

### 编写测试文件

为了保证实现的 SDAA C 算子的正确性，还需要编写测试文件进行验证。通常可以将自定义 SDAA C 算子的计算结果与 `numpy` 或者 `PaddlePaddle` 中功能相同的 api 或 api 组合的计算结果进行比较。这里我们在 `backends/sdaa/tests/unittests` 目录下新建测试文件 `test_custom_cos_op_sdaa.py`，其代码如下：

```python
import numpy as np
import paddle
import paddle_sdaa

x_np = np.arange(20).reshape(4, 5).astype('float32')
out_np = np.cos(x_np)
x_pd = paddle.to_tensor(x_np)
out_pd = paddle_sdaa.sdaa_ext.custom_cos(x_pd)
np.testing.assert_allclose(out_pd, out_np, rtol=1e-6, atol=1e-4)
```

编译、安装 paddle-sdaa 后，在 `backends/sdaa/build` 目录下执行指令 `ctest -V -R test_custom_cos_op_sdaa` 查看测试是否通过。

### 编译、安装、调用

经过以上的几个步骤我们已经完成了需要实现的所有代码。接下来参照 [源码安装指南](../../README.md) 编译、安装即可。安装完成后，可以通过 `paddle_sdaa.sdaa_ext.custom_cos` 调用我们实现的 **cos** 算子：

```python
import paddle
import paddle_sdaa

input = paddle.arange(9.99, dtype='float32')
# input is:
# Tensor(shape=[10], dtype=float32, place=Place(sdaa:0), stop_gradient=True, [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

output1 = paddle.cos(input)
# output1 is:
# Tensor(shape=[10], dtype=float32, place=Place(sdaa:0), stop_gradient=True, [ 1.        ,  0.54030228, -0.41614684, -0.98999250, -0.65364361, 0.28366220,  0.96017027,  0.75390226, -0.14550003, -0.91113025])

output2 = paddle_sdaa.sdaa_ext.custom_cos(input)
# output2 is:
# Tensor(shape=[10], dtype=float32, place=Place(sdaa:0), stop_gradient=True, [ 1.        ,  0.54030228, -0.41614684, -0.98999250, -0.65364361, 0.28366220,  0.96017027,  0.75390226, -0.14550003, -0.91113025])
```

## 小结

总的来说，接入自定义 SDAA C 算子仅需要完成如下几个步骤：

1. 在 `backends/sdaa/sdaac_ops/contrib` 目录下实现 SDAA Kernel
2. 将 SDAA Kernel 中的 host 端调用函数的声明写入 `backends/sdaa/sdaac_ops/contrib/sdaacops_contrib.h` 中
3. 在 `backends/sdaa/sdaa_ext` 目录下实现 PaddlePaddle C++ 算子，在其中调用前面的 SDAA C 算子
4. 在 `backends/sdaa/tests/unittests` 目录下实现测试文件
