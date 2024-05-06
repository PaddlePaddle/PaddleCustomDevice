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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void AclopAnyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& dims,
                    bool keep_dim,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, x, true, out);
    return;
  }
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  // broadcast
  std::vector<int64_t> dim_vec = dims;
  auto x_dims_vec = phi::vectorize(x.dims());
  if (reduce_all) {
    dim_vec.clear();
    for (size_t d = 0; d < x_dims_vec.size(); ++d) {
      dim_vec.push_back(static_cast<int64_t>(d));
    }
  }
  NpuOpRunner runner;
  runner.SetType("ReduceAny")
      .AddInput(x)
      .AddInput(dev_ctx, std::move(dim_vec))
      .AddOutput(*out)
      .AddAttr("keep_dims", keep_dim);
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void AnyKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnAny,
                   (custom_kernel::AclopAnyKernel<T, Context>(
                       dev_ctx, x, dims, keep_dim, out)));
  dev_ctx.template Alloc<T>(out);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, x, true, out);
    return;
  }
  phi::IntArray dim_arr(dims);
  EXEC_NPU_CMD(aclnnAny, dev_ctx, x, dim_arr, keep_dim, *out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    any, npu, ALL_LAYOUT, custom_kernel::AnyKernel, bool) {}
