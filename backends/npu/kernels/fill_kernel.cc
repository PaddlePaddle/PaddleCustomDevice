// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

template <typename T, typename Context>
void FillKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& val) {
  EXEC_NPU_CMD(aclnnInplaceFillScalar, dev_ctx, x, val);
}

template <typename T, typename Context>
void FillGradKernel(const Context& dev_ctx, const phi::DenseTensor& x) {
  EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, x);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fill,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FillKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          bool,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(fill_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FillGradKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          bool,
                          int,
                          int64_t) {}
