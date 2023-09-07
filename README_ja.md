# PaddleCustomDevice

[English](./README_en.md) | [简体中文](./README.md) | 日本語

PaddlePaddle カスタムデバイスの実装。

## ユーザーガイド

プログラム設計ドキュメントは[カスタムデバイスの概要](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_overview_cn.html)を、開発ガイドラインについては、[新しいハードウェアのアクセス例](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_example_cn.html)を、デモコードは [CustomCPU](backends/custom_cpu/README_ja.md) を参照してください。

## ハードウェアバックエンド

PaddleCustomDevice は以下のバックエンドをサポートしています:

- [Ascend NPU 用 PaddlePaddle カスタムデバイス実装](backends/npu/README.md)
- [Cambricon MLU 用 PaddlePaddle カスタムデバイス実装](backends/mlu/README.md)
- [Intel GPU 用 PaddlePaddle カスタムデバイス実装](backends/intel_gpu/README.md)
- [Apple MPS 用 PaddlePaddle カスタムデバイス実装](backends/mps/README.md)
- [Biren GPU 用 PaddlePaddle カスタムデバイス実装](backends/biren_gpu/README.md)

## 著作権とライセンス

PaddleCustomDevice は [Apache-2.0 license](LICENSE) の下で提供されています。
