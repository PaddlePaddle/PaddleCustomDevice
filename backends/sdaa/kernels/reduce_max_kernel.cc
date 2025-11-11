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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {
template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA MaxRawKernel";
  auto dims = axes.GetData();
  std::vector<int64_t> reduce_dims;
  int nDims = x.dims().size();
  if (reduce_all) {
    for (size_t i = 0; i < nDims; i++) {
      reduce_dims.push_back(i);
    }
  } else {
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) {
        reduce_dims.push_back(dims[i] + nDims);
      } else {
        reduce_dims.push_back(dims[i]);
      }
    }
  }

  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doMaxTensor(dev_ctx, x, reduce_dims, out);
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA MaxKernel";
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  custom_kernel::MaxRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(max,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
