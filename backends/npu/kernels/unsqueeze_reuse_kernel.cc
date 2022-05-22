// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/impl/unsqueeze_grad_kernel_impl.h"
#include "paddle/phi/kernels/impl/unsqueeze_kernel_impl.h"
#include "paddle/phi/kernels/unsqueeze_grad_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

using namespace phi;

namespace custom_kernel {}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(
//     unsqueeze, ascend, ALL_LAYOUT, phi::UnsqueezeKernel,
//     bool,
//     int,
//     int8_t,
// // #ifdef PADDLE_WITH_ASCEND_INT64
//     int64_t,
// // #endif
//     float,
//     double) {}

// PD_REGISTER_PLUGIN_KERNEL(unsqueeze_grad,
//                           ascend,
//                           ALL_LAYOUT,
//                           phi::UnsqueezeGradKernel,
//                           bool,
//                           int,
//                           int8_t,
// // #ifdef PADDLE_WITH_ASCEND_INT64
//                           int64_t,
// // #endif
//                           float,
//                           double) {}
