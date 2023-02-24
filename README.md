# PaddleCustomDevice

English | [简体中文](./README_cn.md)

PaddlePaddle custom device implementaion.

## User Guides

To follow up on latest features of custom device in PaddlePaddle, please refer to [Custom Device Support](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/dev_guides/custom_device_docs/index_en.html)

## Hardware Backends

PaddleCustomDevice has supported the following backends:

- [PaddlePaddle Custom Device Implementaion for Ascend NPU](backends/npu/README.md)
- [PaddlePaddle Custom Device Implementaion for Custom CPU](backends/custom_cpu/README.md)
- [PaddlePaddle Custom Device Implementaion for Cambricon MLU](backends/mlu/README.md)

## Environment Variables
| Subject     | Variable Name       | Type   | Description    | Default Value |
| -------- | -------------------------------- | ------ | --------------------------------- | ------------------------------------------------------------ |
| Debug | CUSTOM_DEVICE_BLACK_LIST| String |   Ops in black_list will be fallbacked to CPU  |                                            |

## Copyright and License
PaddleCustomDevice is provided under the [Apache-2.0 license](LICENSE).
