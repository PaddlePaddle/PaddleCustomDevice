# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function, division

import numpy as np
import unittest
import paddle
import paddle_sdaa  # noqa

from paddle.base.layer_helper import LayerHelper

paddle.enable_static()

SEED = 2023

np.random.seed(SEED)


@paddle.jit.to_static(
    input_spec=[
        paddle.static.InputSpec([None, 384, 20, 20], "float32", "input"),
        paddle.static.InputSpec([384, 384, 3, 3], "float32", "filter"),
        paddle.static.InputSpec([384], "float32", "bias"),
        paddle.static.InputSpec([384], "float32", "mean"),
        paddle.static.InputSpec([384], "float32", "scale"),
        paddle.static.InputSpec([384], "float32", "variance"),
    ]
)
def conv_bn_subgraph(input, filter, bias, mean, scale, variance):
    conv_out = paddle.nn.functional.conv2d(
        input,
        filter,
        bias=None,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        data_format="NCHW",
        name=None,
    )

    bn_out = paddle.nn.functional.batch_norm(
        conv_out,
        mean,
        variance,
        scale,
        bias,
        training=False,
        momentum=0.1,
        epsilon=1e-05,
        data_format="NCHW",
    )
    return bn_out


@paddle.jit.to_static(
    input_spec=[
        paddle.static.InputSpec([None, 32], "float32", "x"),
        paddle.static.InputSpec([None, 32], "float32", "y"),
        paddle.static.InputSpec([None, 32], "float32", "z"),
    ]
)
def func(x, y, z):
    return x + y + z


@paddle.jit.to_static(
    input_spec=[
        paddle.static.InputSpec([None, 32], "float32", "x"),
    ]
)
def func_silu(x):
    return x * paddle.nn.functional.sigmoid(x)


@paddle.jit.to_static(
    input_spec=[
        paddle.static.InputSpec([None, 200], "float32", "input"),
        paddle.static.InputSpec([200, 2], "float32", "w"),
        paddle.static.InputSpec([2], "float32", "bias"),
    ]
)
def fc_pass_subgraph(input, w, bias):
    helper = LayerHelper("mul", **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="mul",
        inputs={"X": input, "Y": w},
        outputs={"Out": out},
    )
    return paddle.add(out, bias)


MODLE_FILE = "./saved_model"
MODLE_FILE2 = "./silu_fuse_model"


class TestCustomPass(unittest.TestCase):
    def setUp(self):
        paddle.jit.save(func, MODLE_FILE)

    def test_my_add_n(self):
        config = paddle.inference.Config()
        config.set_prog_file(MODLE_FILE + ".pdmodel")
        config.enable_memory_optim()
        config.enable_custom_device("sdaa")
        pass_builder = config.pass_builder()
        pass_builder.append_pass("custom_add_n")
        print(pass_builder.all_passes())
        predictor = paddle.inference.create_predictor(config)

        np_inputs = [
            np.random.randn(2, 32).astype("float32"),
            np.random.randn(2, 32).astype("float32"),
            np.random.randn(2, 32).astype("float32"),
        ]
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(np_inputs[i])

        predictor.run()
        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        np.testing.assert_allclose(results[0], np.sum(np_inputs, axis=0), rtol=1e-5)


class TestCustomPassSilu(unittest.TestCase):
    def setUp(self):
        paddle.jit.save(func_silu, MODLE_FILE2)

    def test_silu_fuse(self):
        paddle.disable_static()
        config = paddle.inference.Config()
        config.set_prog_file(MODLE_FILE2 + ".pdmodel")
        config.enable_memory_optim()
        config.enable_custom_device("sdaa")
        config.switch_ir_optim(True)
        pass_builder = config.pass_builder()
        pass_builder.append_pass("custom_silu_fuse_pass")
        print(pass_builder.all_passes())
        print("IR Optim is: {}".format(config.ir_optim()))
        predictor = paddle.inference.create_predictor(config)

        np_inputs = [np.random.randn(2, 32).astype("float32")]
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(np_inputs[i])

        predictor.run()
        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        np.testing.assert_allclose(
            results,
            paddle.nn.functional.silu(paddle.to_tensor(np_inputs)).numpy(),
            rtol=1e-5,
        )
        paddle.enable_static()


class TestCustomFcPass(unittest.TestCase):
    def setUp(self):
        self.model_name = "fc"
        paddle.jit.save(fc_pass_subgraph, self.model_name)

        self.batch_size = 64

    def test_custom_fc_n(self):
        config = paddle.inference.Config()
        config.set_prog_file(self.model_name + ".pdmodel")
        config.enable_memory_optim()
        config.enable_custom_device("sdaa")
        pass_builder = config.pass_builder()
        pass_builder.append_pass("custom_fc")
        predictor = paddle.inference.create_predictor(config)

        np_inputs = [
            np.random.randn(self.batch_size, 200).astype("float32"),
            np.random.randn(200, 2).astype("float32"),
            np.random.randn(2).astype("float32"),
        ]
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(np_inputs[i])

        predictor.run()
        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            cpu_output = paddle._legacy_C_ops.fc(
                paddle.to_tensor(np_inputs[0]),
                paddle.to_tensor(np_inputs[1]),
                paddle.to_tensor(np_inputs[2]),
                "activation_type",
                "",
                "in_num_col_dims",
                1,
            )

        np.testing.assert_allclose(results[0], cpu_output.numpy(), rtol=1e-4, atol=1e-2)


class TestCustomConvBnFusedPass(unittest.TestCase):
    def setUp(self):
        self.model_name = "conv_bn_subgraph"
        paddle.jit.save(conv_bn_subgraph, self.model_name)

        self.batch_size = 32

    def _get_output_with_place(self, np_inputs, place):
        config = paddle.inference.Config()
        config.set_prog_file(self.model_name + ".pdmodel")
        config.enable_memory_optim()

        if isinstance(place, paddle.CustomPlace):
            config.enable_custom_device("sdaa")
            pass_builder = config.pass_builder()
            pass_builder.append_pass("custom_conv_bn_fuse_pass")
        else:
            pass_builder = config.pass_builder()

            for pass_name in pass_builder.all_passes():
                pass_builder.delete_pass(pass_name)

            # pass_builder.append_pass("conv_bn_fuse_pass")

        config.switch_ir_debug(True)
        predictor = paddle.inference.create_predictor(config)

        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(np_inputs[i])

        predictor.run()

        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        predictor.clear_intermediate_tensor()
        return results

    def test_custom_conv_bn_fused(self):
        np_inputs = [
            np.random.randn(self.batch_size, 384, 20, 20).astype("float32"),
            np.random.randn(384, 384, 3, 3).astype("float32") * 0.001,
            np.random.randn(384).astype("float32"),
            np.zeros(384).astype("float32"),
            np.random.randn(384).astype("float32"),
            np.ones(shape=[384]).astype("float32"),
        ]

        cpu_result = self._get_output_with_place(np_inputs, paddle.CPUPlace())
        sdaa_result = self._get_output_with_place(
            np_inputs, paddle.CustomPlace("sdaa", 0)
        )
        np.testing.assert_allclose(cpu_result[0], sdaa_result[0], atol=1e-2)


if __name__ == "__main__":
    unittest.main()
