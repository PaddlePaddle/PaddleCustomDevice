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

#include <cmath>
#include <cstring>

#include "funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void logic_kernel_impl(const Context& ctx,
                       const phi::DenseTensor& x,
                       const std::vector<int64_t>& dims,
                       phi::DenseTensor* out,
                       TensorLogicType tlt) {
  VLOG(4) << "Call SDAA LogicKernel";

  ctx.template Alloc<T>(out);

  // support 0D tensor
  if (x.dims().size() == 0) {
    phi::Copy(ctx, x, ctx.GetPlace(), false, out);
    return;
  }

  sdaa_ops::doLogicTensor(ctx, x, dims, tlt, out);
}

#define DEFINE_TECODNN_LOGIC_KERNEL(logic_kernel, logic_type)         \
  template <typename T, typename Context>                             \
  void logic_kernel(const Context& dev_ctx,                           \
                    const phi::DenseTensor& x,                        \
                    const std::vector<int64_t>& dims,                 \
                    bool keep_dim,                                    \
                    phi::DenseTensor* out) {                          \
    logic_kernel_impl<T, Context>(dev_ctx, x, dims, out, logic_type); \
  }
DEFINE_TECODNN_LOGIC_KERNEL(AllKernel, TensorLogicType::all);
DEFINE_TECODNN_LOGIC_KERNEL(AnyKernel, TensorLogicType::any);
#undef DEFINE_TECODNN_LOGIC_KERNEL

#define DEFINE_TECODNN_LOGIC_RAW_KERNEL(logic_raw_kernel, logic_type) \
  template <typename T, typename Context>                             \
  void logic_raw_kernel(const Context& dev_ctx,                       \
                        const phi::DenseTensor& x,                    \
                        const std::vector<int64_t>& dims,             \
                        bool keep_dim,                                \
                        bool reduce_all,                              \
                        phi::DenseTensor* out) {                      \
    logic_kernel_impl<T, Context>(dev_ctx, x, dims, out, logic_type); \
  }
DEFINE_TECODNN_LOGIC_RAW_KERNEL(AllRawKernel, TensorLogicType::all);
DEFINE_TECODNN_LOGIC_RAW_KERNEL(AnyRawKernel, TensorLogicType::any);
#undef DEFINE_TECODNN_LOGIC_RAW_KERNEL

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    all, sdaa, ALL_LAYOUT, custom_kernel::AllKernel, bool) {}

PD_REGISTER_PLUGIN_KERNEL(
    all_raw, sdaa, ALL_LAYOUT, custom_kernel::AllRawKernel, bool) {}

PD_REGISTER_PLUGIN_KERNEL(
    any_raw, sdaa, ALL_LAYOUT, custom_kernel::AnyRawKernel, bool) {}

PD_REGISTER_PLUGIN_KERNEL(
    any, sdaa, ALL_LAYOUT, custom_kernel::AnyKernel, bool) {}
