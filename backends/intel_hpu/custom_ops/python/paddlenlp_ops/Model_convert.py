# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import paddle
from safetensors.paddle import load_file, save_file
from tqdm import tqdm
import json
import shutil
import glob
from typing import Dict

paddle.device.set_device("intel_hpu:5")

MAX_FILE_SIZE_IN_GB = 5
max_size_bytes = MAX_FILE_SIZE_IN_GB * 1000**3


def tensor_size(tensor):
    return tensor.nbytes if hasattr(tensor, "nbytes") else tensor.numpy().nbytes


def tensors_total_size(tensors_dict):
    return sum(tensor_size(tensor) for tensor in tensors_dict.values())


def save_tail_tensors_and_index(
    tensors_dict,
    measurement_files,
    model_fp8_path,
    total_size,
    out_file_idx,
    out_files,
    approximate_total_files,
):
    for measurement_file in measurement_files:
        with open(measurement_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split("\t")
                if value == 0.0:
                    print(f"warning: amax is 0.0 for {key}, set to 1e-5")
                    value = 1e-5
                if "self_attn" not in key:
                    scale = float(value) / 240.0
                else:
                    scale = float(value)
                meas_scale_tensor = paddle.to_tensor([scale], dtype=paddle.bfloat16)
                # print(f"--- meas_scale for {key}: {meas_scale_tensor} ---")
                if key in tensors_dict:
                    tensors_dict[key] = paddle.maximum(
                        tensors_dict[key], meas_scale_tensor
                    )
                else:
                    tensors_dict[key] = meas_scale_tensor
                    total_size += tensor_size(meas_scale_tensor)

    file_name = f"model-{out_file_idx:05d}-of-{approximate_total_files:05d}.safetensors"
    file_path = os.path.join(model_fp8_path, file_name)
    save_file(tensors_dict, file_path)
    out_files.append({"filename": file_name, "keys": list(tensors_dict.keys())})

    index_json = {"metadata": {"total_size": total_size}, "weight_map": {}}
    for file_info in out_files:
        for key in file_info["keys"]:
            index_json["weight_map"][key] = file_info["filename"]

    index_path = os.path.join(model_fp8_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_json, f, indent=2)


def tensorwise_quant_to_fp8(tensor):
    x_abs = paddle.abs(tensor).astype(paddle.float32)
    x_amax = paddle.amax(x_abs)
    x_amax = paddle.clip(x_amax, min=1e-8)
    scale = x_amax / 240.0
    x_scaled = (tensor.cast("float32") / scale).cast("float8_e4m3fn").clone()

    return paddle.view(x_scaled, "int8").clone(), paddle.to_tensor([scale]).cast(
        "bfloat16"
    )


def channelwise_quant_to_fp8(tensor):
    # Channel-wise quantization along the last dimension (N)
    x_abs = paddle.abs(tensor).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=0)  # shape: [N]
    x_amax = paddle.clip(x_amax, min=1e-8)
    scale = x_amax / 240.0  # shape: [N]
    x_scaled = (
        (tensor.cast("float32") / scale.cast("float32")).cast("float8_e4m3fn").clone()
    )

    return paddle.view(x_scaled, "int8").clone(), scale.cast("bfloat16").clone()


def process_safetensors_file(
    tensors_dict,
    src_path,
    model_fp8_path,
    total_size,
    out_file_idx,
    out_files,
    max_size_bytes,
    approximate_total_files,
):
    current_size = tensors_total_size(tensors_dict)

    loaded_tensors = load_file(src_path)
    for key, tensor in loaded_tensors.items():
        if "_proj.weight" in key:
            if tensor.dtype != paddle.bfloat16:
                print(
                    f"Warning: Expected bfloat16 tensor for key {key}, but got {tensor.dtype}. Skipping."
                )
                continue
            else:
                tensor = paddle.Tensor(tensor, zero_copy=True)
            if ".experts." in key:  # except for shared_experts
                quant_tensor, scale = channelwise_quant_to_fp8(tensor)
            else:
                quant_tensor, scale = tensorwise_quant_to_fp8(tensor)

            t_size = tensor_size(quant_tensor) + tensor_size(scale)
            if current_size + t_size > max_size_bytes and tensors_dict:
                file_name = f"model-{out_file_idx:05d}-of-{approximate_total_files:05d}.safetensors"
                file_path = os.path.join(model_fp8_path, file_name)
                save_file(tensors_dict, file_path)
                out_files.append(
                    {"filename": file_name, "keys": list(tensors_dict.keys())}
                )
                out_file_idx += 1
                tensors_dict = {}
                current_size = 0

            new_key = key.replace("_proj.weight", "_proj.quant_weight")
            tensors_dict[new_key] = quant_tensor
            scale_key = key.replace("_proj.weight", "_proj.weight_scale")
            tensors_dict[scale_key] = scale
            current_size += t_size
            total_size += t_size
        else:
            t_size = tensor_size(tensor)
            if current_size + t_size > max_size_bytes and tensors_dict:
                file_name = f"model-{out_file_idx:05d}-of-{approximate_total_files:05d}.safetensors"
                file_path = os.path.join(model_fp8_path, file_name)
                save_file(tensors_dict, file_path)
                out_files.append(
                    {"filename": file_name, "keys": list(tensors_dict.keys())}
                )
                out_file_idx += 1
                tensors_dict = {}
                current_size = 0
            tensors_dict[key] = tensor
            current_size += t_size
            total_size += t_size
    return tensors_dict, total_size, out_file_idx, out_files


def main():
    print(
        f"Usage: python {sys.argv[0]} [model_bf16_path] [model_fp8_path] [model_measurement_file_or_folder] <ranks_total_number>"
    )
    if len(sys.argv) > 3:
        model_bf16_path = sys.argv[1]
        model_fp8_path = sys.argv[2]
        model_measurement_file = sys.argv[3]
        ranks = "0"
    if len(sys.argv) > 4:
        ranks = sys.argv[4]
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Error: Invalid number of arguments.")
        return
    os.makedirs(model_fp8_path, exist_ok=True)

    if os.path.isdir(model_measurement_file):
        measurement_files = [
            os.path.join(model_measurement_file, f)
            for f in os.listdir(model_measurement_file)
            if os.path.isfile(os.path.join(model_measurement_file, f))
        ]
    elif ranks.isdigit() and int(ranks) > 1:
        measurement_files = [
            f"{os.path.splitext(model_measurement_file)[0]}_{i}{os.path.splitext(model_measurement_file)[1]}"
            for i in range(int(ranks))
        ]
    else:
        measurement_files = [model_measurement_file]

    for measurement_file in measurement_files:
        if not os.path.isfile(measurement_file):
            print(f"Error: Measurement file not found: {measurement_file}")
            return

    # copy none safetensor files (except model.safetensors.index.json) to new folder
    for item_name in os.listdir(model_bf16_path):
        source_path = os.path.join(model_bf16_path, item_name)
        if os.path.isfile(source_path):
            if item_name == "model.safetensors.index.json":
                with open(source_path, "r") as f:
                    index_data = json.load(f)
                    total_size = index_data.get("metadata", {}).get("total_size", None)
            elif item_name == "config.json":
                with open(source_path, "r") as f:
                    config_data = json.load(f)
                config_data["quantization_config"] = {
                    "dense_quant_type": "tensor_wise_fp8",
                    "moe_quant_type": "tensor_wise_fp8",
                    "quantization": "mix_quant",
                    "kv_cache_quant_type": "float8_e4m3",
                    "is_quantized": True,
                }
                destination_path = os.path.join(model_fp8_path, item_name)
                with open(destination_path, "w") as f:
                    json.dump(config_data, f, indent=2)
            elif not item_name.lower().endswith(".safetensors"):
                destination_path = os.path.join(model_fp8_path, item_name)
                try:
                    shutil.copy2(source_path, destination_path)
                except Exception as e:
                    print(f"Error copying {item_name}: {e}")

    # 计算预计总文件数
    total_size *= 0.506
    approximate_total_files = int((total_size + max_size_bytes - 1) // max_size_bytes)
    print(f"Approximate total files to be generated: {approximate_total_files}")
    total_size = 0
    out_file_idx = 1
    tensors_dict: Dict[str, paddle.Tensor] = {}
    out_files = []

    search_pattern = os.path.join(model_bf16_path, "*.safetensors")
    safetensor_files = glob.glob(search_pattern)

    if not safetensor_files:
        print("Warning: No *.safetensors files found in the source directory.")
        return

    for file in tqdm(
        safetensor_files,
        desc=f"Loading safetensor files from {model_bf16_path}",
        unit="file",
    ):
        (tensors_dict, total_size, out_file_idx, out_files,) = process_safetensors_file(
            tensors_dict,
            file,
            model_fp8_path,
            total_size=total_size,
            out_file_idx=out_file_idx,
            out_files=out_files,
            max_size_bytes=max_size_bytes,
            approximate_total_files=approximate_total_files,
        )

    save_tail_tensors_and_index(
        tensors_dict,
        measurement_files,
        model_fp8_path,
        total_size,
        out_file_idx,
        out_files,
        approximate_total_files,
    )


main()
