---
comments: true
---

# PaddleX模型列表（燧原 GCU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。


## [行人检测模块](../module_usage/tutorials/cv_modules/human_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE-L_human</td>
<td>48.0</td>
<td>196.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE-L_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_human_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE-S_human</td>
<td>42.5</td>
<td>28.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE-S_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_human_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[CrowdHuman](https://bj.bcebos.com/v1/paddledet/data/crowdhuman.zip)<b>验证集 mAP(0.5:0.95)。</b>

## [文本检测模块](../module_usage/tutorials/ocr_modules/text_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>检测Hmean（%）</th>
<th>模型存储大小（M）</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.79</td>
<td>4.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.69</td>
<td>100.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。</b>

## [文本识别模块](../module_usage/tutorials/ocr_modules/text_recognition.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>识别Avg Accuracy(%)</th>
<th>模型存储大小（M）</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.20</td>
<td>10.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td>79.20</td>
<td>71.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。</b>