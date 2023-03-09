# PaddleCustomDevice

简体中文 | [English](./README_en.md)

『飞桨』自定义硬件接入实现。

## 使用指南

方案设计参考[Custom Device 接入方案介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_overview_cn.html)，开发指南请参考[新硬件接入示例](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_example_cn.html)且示例代码位于 [CustomCPU](backends/custom_cpu/README_cn.md)。

## 硬件后端

飞桨自定义硬件接入支持如下硬件后端：

- [飞桨自定义接入硬件后端(昇腾NPU)](backends/npu/README_cn.md)
- [飞桨自定义接入硬件后端(寒武纪MLU)](backends/mlu/README_cn.md)
- [飞桨自定义接入硬件后端(英特尔GPU)](backends/intel_gpu/README.md)

## 版权和许可证

PaddleCustomDevice由[Apache-2.0 license](LICENSE)提供。
