from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

from .llama_pass import replace_embeding, llama_fuse_attention_dynamic_layer1, llama_fuse_attention_dynamic_layer2, llama_fuse_attention_dynamic_first_parallel_layer, llama_fuse_attention_dynamic_parallel_layer, llama_lmhead, llama_fuse_attention_dynamic_first_parallel_layer_be61d, llama_fuse_attention_dynamic_parallel_layer_be61d
from .remove_pass import remove_fused_bias_residual_layernorm, remove_rebuild_padding, remove_get_padding_offset, remove_get_token_penalty_multi_scores, save_with_output_delay_pass_1st, save_with_output_delay_pass_2nd
from .paged_attention_quant_pass import (llama_fuse_attention_dynamic_parallel_first, llama_fuse_attention_dynamic_parallel_others, remove_fused_bias_residual_layernorm_quant, llama_lmhead_quant, llama_fuse_attention_dynamic_parallel_end, 
                                         llama_fuse_attention_dynamic_parallel_first_65B, llama_fuse_attention_dynamic_parallel_others_65B,
                                         llama_fuse_attention_dynamic_smooth_parallel_first_65B, llama_fuse_attention_dynamic_smooth_parallel_others_65B)

def setUp():
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )

def register_pass(pass_builder, pass_name):
    pass_builder.append_pass(pass_name)
    paddle.base.core.register_subgraph_pass(pass_name)

def addPasses(pass_builder, model_type):
    if model_type == "llama7B_mp8_dynamic_batch":
        register_pass(pass_builder, "llama_fuse_attention_dynamic_first_parallel_layer_be61d")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_layer_be61d")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_layer1")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_layer2")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_first_parallel_layer")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_layer")
        register_pass(pass_builder, "remove_fused_bias_residual_layernorm")
        register_pass(pass_builder, "llama_lmhead")
        register_pass(pass_builder, "save_with_output_delay_pass_2nd")
        register_pass(pass_builder, "save_with_output_delay_pass_1st")
        register_pass(pass_builder, "replace_embeding")
    elif model_type == "llama65B_mp8_dynamic_batch":
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_layer1")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_layer2")
        register_pass(pass_builder, "llama65B_fuse_attention_dynamic_first_parallel_layer")
        register_pass(pass_builder, "llama65B_fuse_attention_dynamic_parallel_layer")
        register_pass(pass_builder, "remove_fused_bias_residual_layernorm")
        register_pass(pass_builder, "llama_lmhead")
        register_pass(pass_builder, "save_with_output_delay_pass_2nd")
        register_pass(pass_builder, "save_with_output_delay_pass_1st")
        register_pass(pass_builder, "replace_embeding")
    elif model_type == "llama13B_mp2":
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_first")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_end")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_others")
        register_pass(pass_builder, "llama_lmhead_quant")
    elif model_type == "llama65B_mp8":
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_first_65B")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_parallel_others_65B")
        register_pass(pass_builder, "remove_fused_bias_residual_layernorm")
        register_pass(pass_builder, "llama_lmhead_quant")
    elif model_type == "llama65B_mp8_smooth":
        register_pass(pass_builder, "llama_fuse_attention_dynamic_smooth_parallel_first_65B")
        register_pass(pass_builder, "llama_fuse_attention_dynamic_smooth_parallel_others_65B")
        register_pass(pass_builder, "remove_fused_bias_residual_layernorm")
        register_pass(pass_builder, "llama_lmhead_quant")
    else:
        print("NPU pass not support")
