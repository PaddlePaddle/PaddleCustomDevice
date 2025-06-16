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
#include "kernels/cuda_kernels/weight_quantize_kernel_gpu_impl.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void WeightQuantizeKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::string& algo,
                          const int32_t arch,
                          const int32_t group_size,
                          DenseTensor* out,
                          DenseTensor* scale) {
  PADDLE_ENFORCE_EQ(
      (group_size == -1),
      true,
      common::errors::InvalidArgument(
          "Currently, group_size only support -1(per-channel), 64 or 128."));

  dev_ctx.template Alloc<int8_t>(out);
  dev_ctx.template Alloc<T>(scale);
  std::vector<int> weight_shape{static_cast<int>(x.dims()[0]),
                                static_cast<int>(x.dims()[1])};
  if (algo == "weight_only_int8") {
    weight_quant_gpu<T, Context>(dev_ctx,
                                 x.data<T>(),
                                 out->data<int8_t>(),
                                 scale->data<T>(),
                                 weight_shape,
                                 arch,
                                 algo);
  } else {
    PADDLE_FATAL("The algo must be in ['weight_only_int8',], but got[%s]",
                 algo);
  }
}
}  // namespace phi

PD_REGISTER_PLUGIN_KERNEL(weight_quantize,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::WeightQuantizeKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
