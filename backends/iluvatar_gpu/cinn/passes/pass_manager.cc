// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

#include "paddle/phi/backends/device_ext.h"

namespace paddle {
namespace custom_device {
namespace iluvatar {

// 负责应用自定义的图优化 Pass
// 目前阶段先留空，直接返回成功
C_Status IluvatarApplyCustomPass(void* dev_ptr, void* ir_module) {
  // VLOG(0) << "[Iluvatar] IluvatarApplyCustomPass called (No-op)";
  return C_Status::C_SUCCESS;
}

}  // namespace iluvatar
}  // namespace custom_device
}  // namespace paddle
