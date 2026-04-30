// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

// 引入 Paddle 定义的 C 接口结构体
#include "paddle/phi/backends/device_ext.h"

namespace paddle {
namespace custom_device {
namespace iluvatar {

/**
 * @brief 初始化 CINN 接口
 * * 这个函数由 runtime.cc 中的 InitPlugin 调用。
 * 它负责将 iluvatar_gpu/cinn 下实现的编译器和运行时函数指针，
 * 填充到 device_interface->cinn_interface 中。
 * * @param device_interface Paddle Host 侧传入的设备接口指针
 */
void InitCinnInterface(C_DeviceInterface* device_interface);

}  // namespace iluvatar
}  // namespace custom_device
}  // namespace paddle
