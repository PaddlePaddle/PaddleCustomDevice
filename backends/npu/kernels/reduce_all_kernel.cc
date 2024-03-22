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
void AclopAllRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const std::vector<int64_t>& dims,
                       bool keep_dim,
                       bool reduce_all,
                       phi::DenseTensor* out) {
  bool reduce_all_f = dims.size() == 0 ||
                      static_cast<int>(dims.size()) == x.dims().size() ||
                      reduce_all;
  std::vector<int64_t> dims_vec = dims;
  dev_ctx.template Alloc<T>(out);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, x, true, out);
    return;
  }
  if (reduce_all_f) {
    dims_vec.clear();
    for (size_t i = 0; i < x.dims().size(); ++i) {
      dims_vec.push_back(static_cast<int64_t>(i));
    }
  }
  auto stream = dev_ctx.stream();
  NpuOpRunner runner;
  runner.SetType("ReduceAll")
      .AddInput(x)
      .AddInput(dev_ctx, std::move(dims_vec))
      .AddOutput(*out)
      .AddAttr("keep_dims", keep_dim);
  runner.Run(stream);
}

template <typename T, typename Context>
void AllRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnAll,
                   (custom_kernel::AclopAllRawKernel<T, Context>(
                       dev_ctx, x, dims, keep_dim, reduce_all, out)));
  if (x.storage_properties_initialized()) {
    custom_kernel::AclopAllRawKernel<T, Context>(
        dev_ctx, x, dims, keep_dim, reduce_all, out);
    return;
  }
  bool reduce_all_f = dims.size() == 0 ||
                      static_cast<int>(dims.size()) == x.dims().size() ||
                      reduce_all;
  std::vector<int64_t> dims_vec = dims;
  dev_ctx.template Alloc<T>(out);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, x, true, out);
    return;
  }
  if (reduce_all_f) {
    dims_vec.clear();
    for (size_t i = 0; i < x.dims().size(); ++i) {
      dims_vec.push_back(static_cast<int64_t>(i));
    }
  }
  EXEC_NPU_CMD(aclnnAll, dev_ctx, x, dims_vec, keep_dim, *out);
}

template <typename T, typename Context>
void AllKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all =
      dims.size() == 0 || static_cast<int>(dims.size()) == x.dims().size();
  custom_kernel::AllRawKernel<T, Context>(
      dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    all, npu, ALL_LAYOUT, custom_kernel::AllKernel, bool) {}

PD_REGISTER_PLUGIN_KERNEL(
    all_raw, npu, ALL_LAYOUT, custom_kernel::AllRawKernel, bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
