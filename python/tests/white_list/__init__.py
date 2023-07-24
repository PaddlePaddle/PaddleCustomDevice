#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

# TODO(duanyanhui):
# Devices differ in accurary, so we need to build different white_list for
# diffrent device. For example, ascend dose not aupport vell well for int64_t
# and double. The cast of data type will bring errors. We need to put that
# kernel in the op_threshlod_white_list.

# Next, we will built white_list for each device and put it on backends.
