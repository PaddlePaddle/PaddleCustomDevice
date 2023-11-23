// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/cast_kernel.h"

#include "kernels/funcs/gcu_op_runner.h"

PD_REGISTER_PLUGIN_KERNEL(cast,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GcuCastKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
