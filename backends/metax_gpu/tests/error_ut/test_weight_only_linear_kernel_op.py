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
import paddle

from paddle.nn.quant import weight_only_linear, weight_quantize

for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )


def print_tensor_stats(name, tensor, max_elements=30):
    """Helper function to print tensor stats and partial data"""
    print(f"\n{name} Tensor Info:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Values (first {min(max_elements, tensor.size)} elements):")

    # Flatten and convert to numpy for printing
    flat_data = tensor.flatten().numpy()
    print(flat_data[: min(max_elements, tensor.size)])


def testWeightOnlyLinear():
    _cases = [([2, 2048], [8192, 2048], [8192]), ([2, 8192], [2560, 8192], [2560])]

    errors = []
    for x_shape, weight_shape, scale_shape in _cases:
        try:
            print(
                "testWeightOnlyLinear with x_shape={}, weight_shape={}, scale_shape={}".format(
                    x_shape, weight_shape, scale_shape
                )
            )
            x = paddle.cast(paddle.randn(x_shape), dtype="float16")
            print_tensor_stats("Input x", x)
            weight = paddle.cast(paddle.randint(0, 127, weight_shape), dtype="int8")
            print_tensor_stats("Weight", weight)
            scale = paddle.randn(scale_shape, dtype="float16")
            print_tensor_stats("Scale", scale)
            bias = paddle.cast(paddle.randn(scale_shape), dtype="float16")
            print_tensor_stats("Bias", bias)
            output = weight_only_linear(
                x, weight, bias=bias, weight_scale=scale, weight_dtype="int8"
            )
            print_tensor_stats("Output", output)

            if output.shape != [x_shape[0], weight_shape[0]]:
                errors.append(
                    f"Shape mismatch: Expected {[x_shape[0], weight_shape[0]]}, "
                    f"got {output.shape} with x_shape={x_shape}, weight_shape={weight_shape}"
                )

        except Exception as e:
            errors.append(
                f"Failed with x_shape={x_shape}, weight_shape={weight_shape}: {str(e)}"
            )

    if errors:
        raise AssertionError(
            "\n".join(
                [
                    f"{len(errors)} errors occurred:",
                    *[f"  {i+1}. {err}" for i, err in enumerate(errors)],
                ]
            )
        )
    else:
        print("testWeightOnlyLinear Passed!")


def testWeightQuantize():
    _cases = [[64, 32], [128, 64], [256, 128]]
    errors = []
    for idx, shape in enumerate(_cases):
        try:
            print("\n" + "=" * 50)
            print(f"testWeightQuantize Case #{idx+1}: Input shape={shape}")

            paddle.seed(2023)
            x = paddle.rand(shape=shape, dtype="float16")
            print_tensor_stats("Input x", x)

            quant_weight, scale = weight_quantize(x, algo="weight_only_int8", arch=80)

            print_tensor_stats("Quantized Weight", quant_weight)
            print_tensor_stats("Quantization Scale", scale)

            expected_weight_shape = [shape[1], shape[0]]
            expected_scale_shape = [shape[1]]

            if quant_weight.shape != expected_weight_shape:
                errors.append(
                    f"Case #{idx+1}: Weight shape error. Expected {expected_weight_shape}, got {quant_weight.shape}"
                )
            if scale.shape != expected_scale_shape:
                errors.append(
                    f"Case #{idx+1}: Scale shape error. Expected {expected_scale_shape}, got {scale.shape}"
                )

        except Exception as e:
            errors.append(f"Case #{idx+1} failed: {str(e)}")

    if errors:
        raise AssertionError(
            "\n".join(
                [
                    f"{len(errors)} errors in testWeightQuantize:",
                    *[f"  {err}" for err in errors],
                ]
            )
        )
    else:
        print("testWeightQuantize Passed!")


if __name__ == "__main__":
    testWeightOnlyLinear()
    testWeightQuantize()
