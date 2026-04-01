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

#pragma once

// Include the Paddle-defined C interface structures
#include "paddle/phi/backends/device_ext.h"

namespace paddle {
namespace custom_device {
namespace metax {

/**
 * @brief Initialize the CINN interface.
 *
 * This function is called by InitPlugin in runtime.cc.
 * It populates device_interface->cinn_interface with the compiler
 * and runtime function pointers implemented under metax_gpu/cinn.
 *
 * @param device_interface The device interface pointer passed from the Paddle
 * host side.
 */
void InitCinnInterface(C_DeviceInterface* device_interface);

}  // namespace metax
}  // namespace custom_device
}  // namespace paddle
