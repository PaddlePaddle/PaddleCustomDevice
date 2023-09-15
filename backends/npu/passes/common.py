from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

paddle.enable_static()

def setUp():
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )

def register_pass(pass_builder, pass_name):
    pass_builder.append_pass(pass_name)
    paddle.fluid.core.register_subgraph_pass(pass_name)

def addPasses(pass_builder, model_type):
    if model_type == "llama_mp8_dynamic_batch":
        register_pass(pass_builder, "llama_fuse_attention_dynamic_layer1")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_layer2")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_first_parallel_layer")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_layer")
        register_pass(pass_builder, "remove_fused_bias_residual_layernorm")
        register_pass(pass_builder, "remove_rebuild_padding")
        register_pass(pass_builder, "remove_get_padding_offset")
    else:
        print("NPU pass not support")