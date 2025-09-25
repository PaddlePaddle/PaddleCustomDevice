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

#include "custom_op/custom_op_common.h"

std::vector<paddle::Tensor> GetDeviceTypeKernel() {
  PADDLE_GCU_KERNEL_TRACE("get_device_type_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: get_device_type_gcu";
  int32_t chip_type = -1;
  if (custom_kernel::IsScorpio()) {
    chip_type = 3;
  } else if (custom_kernel::IsLibra()) {
    chip_type = 4;
  } else {
    phi::errors::Unavailable("Not supported device: %s.",
                             custom_kernel::GetTargetName().c_str());
  }
  auto device_type =
      paddle::full({1}, chip_type, paddle::DataType::INT32, paddle::CPUPlace());
  return {device_type};
}

PD_BUILD_OP(get_device_type_gcu)
    .Outputs({"device_type"})
    .SetKernelFn(PD_KERNEL(GetDeviceTypeKernel));
