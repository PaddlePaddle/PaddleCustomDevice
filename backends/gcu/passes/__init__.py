#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from .common import setUp

from .gcu_conv_add_mul_add_hard_swish_mul_add_fuse import (
    gcu_fuse_conv_add_mul_add_hard_swish_mul_add,
    gcu_fuse_depthwise_conv_add_mul_add_hard_swish_mul_add,
    gcu_fuse_depthwise_conv_add_mul_add,
)
from .gcu_conv_bias_activate_fuse import (
    gcu_fuse_conv_bias,
    gcu_fuse_conv_bias_activate,
)
from .gcu_conv_bn_fuse import (
    gcu_fuse_conv_bn,
    gcu_fuse_conv_bn_swish,
    gcu_fuse_conv_bn_relu,
    gcu_fuse_conv_bn_hard_swish,
)

from .gcu_conv_bn_hard_swish_fuse import (
    gcu_fuse_depthwise_conv_bn_hard_swish,
)

from .gcu_dot_bias_fuse import (
    gcu_fuse_dot_bias,
)

from .gcu_mul_add_fuse import (
    gcu_fuse_mul_add,
)

from .common import register_pass
