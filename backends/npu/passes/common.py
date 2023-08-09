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

import os
import paddle


paddle.enable_static()


def setUp():
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )


def addPasses(pass_builder):
    pass_builder.append_pass("generate_ffn")
    pass_builder.append_pass("generate_add_norm")
    pass_builder.append_pass("generate_linear")
    pass_builder.append_pass("generate_pad2d")
    paddle.fluid.core.register_subgraph_pass("generate_ffn")
    paddle.fluid.core.register_subgraph_pass("generate_add_norm")
    paddle.fluid.core.register_subgraph_pass("generate_linear")
    paddle.fluid.core.register_subgraph_pass("generate_pad2d")
