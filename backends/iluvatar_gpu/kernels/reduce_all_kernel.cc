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

#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "glog/logging.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

using complex64 = ::phi::dtype::complex<float>;

namespace phi {

template <typename T, typename Context>
void AllKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out) {
  auto x_dim = x.dims();
  for (int i = 0; i < x_dim.size(); i++) {
    PADDLE_ENFORCE_LE(
        0,
        x_dim[i],
        errors::InvalidArgument(
            "The dims of Input(X) should be greater than or equal to 0."));
  }
  if (x.numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    if (out->numel() > 0) {
      std::vector<int64_t> vec_dims = common::vectorize(out->dims());
      phi::Full<bool, Context>(dev_ctx, phi::IntArray(vec_dims), 1, out);
    }
    return;
  }
  bool reduce_all = recompute_reduce_all(x, dims);
  AllRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(all,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AllKernel,
                          float,
                          int,
                          int64_t,
                          bool,
                          complex64) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}