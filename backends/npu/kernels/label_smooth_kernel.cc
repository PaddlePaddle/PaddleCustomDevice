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
void AclopLabelSmoothMuls(const Context& place,
                          const phi::DenseTensor* in,
                          float val,
                          phi::DenseTensor* out) {
  out->Resize(in->dims());
  place.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Muls", {*in}, {*out}, {{"value", val}});
  runner.Run(place.stream());
}

template <typename T, typename Context>
void LabelSmoothMuls(const Context& place,
                     const phi::DenseTensor* in,
                     float val,
                     phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnMuls,
      (custom_kernel::AclopLabelSmoothMuls<T, Context>(place, in, val, out)));
  out->Resize(in->dims());
  place.template Alloc<T>(out);
  phi::Scalar value(val);
  EXEC_NPU_CMD(aclnnMuls, place, *in, value, *out);
}

template <typename T, typename Context>
void AclopLabelSmoothAdds(const Context& place,
                          const phi::DenseTensor* in,
                          float val,
                          phi::DenseTensor* out) {
  out->Resize(in->dims());
  place.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Adds", {*in}, {*out}, {{"value", val}});
  runner.Run(place.stream());
}

template <typename T, typename Context>
void LabelSmoothAdds(const Context& place,
                     const phi::DenseTensor* in,
                     float val,
                     phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnAdds,
      (custom_kernel::AclopLabelSmoothAdds<T, Context>(place, in, val, out)));
  out->Resize(in->dims());
  place.template Alloc<T>(out);
  phi::Scalar value(val);
  phi::Scalar alpha(1.0f);
  EXEC_NPU_CMD(aclnnAdds, place, *in, value, alpha, *out);
}

template <typename T, typename Context>
void AclopLabelSmoothAddBroadCast(const Context& place,
                                  const phi::DenseTensor* in1,
                                  const phi::DenseTensor* in2,
                                  phi::DenseTensor* out) {
  place.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("AddV2", {*in1, *in2}, {*out}, {});
  runner.Run(place.stream());
}

template <typename T, typename Context>
void LabelSmoothAddBroadCast(const Context& place,
                             const phi::DenseTensor* in1,
                             const phi::DenseTensor* in2,
                             phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnAdd,
                   (custom_kernel::AclopLabelSmoothAddBroadCast<T, Context>(
                       place, in1, in2, out)));

  phi::Scalar alpha = 1;
  if (in1->dtype() == phi::DataType::FLOAT32 &&
      (in2->dtype() == phi::DataType::BFLOAT16 ||
       in2->dtype() == phi::DataType::FLOAT16)) {
    place.template Alloc<float>(out);
  } else {
    place.template Alloc<T>(out);
  }

  EXEC_NPU_CMD(aclnnAdd, place, *in1, *in2, alpha, *out);
}

template <typename T, typename Context>
void LabelSmoothKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const paddle::optional<phi::DenseTensor>& dist,
                       float epsilon,
                       phi::DenseTensor* out) {
  auto label_dim = x.dims()[x.dims().size() - 1];
  auto stream = dev_ctx.stream();

  if (dist) {
    phi::DenseTensor tmp;
    phi::DenseTensor tmp2;
    auto& dist_tensor = *dist.get_ptr();
    LabelSmoothMuls<T, Context>(dev_ctx, &x, (1 - epsilon), &tmp);
    LabelSmoothMuls<T, Context>(dev_ctx, &dist_tensor, epsilon, &tmp2);
    tmp2.Resize({1, label_dim});
    LabelSmoothAddBroadCast<T, Context>(dev_ctx, &tmp, &tmp2, out);
  } else {
    phi::DenseTensor tmp;
    LabelSmoothMuls<T, Context>(dev_ctx, &x, (1 - epsilon), &tmp);
    LabelSmoothAdds<T, Context>(dev_ctx, &tmp, (epsilon / label_dim), out);
  }
}

template <typename T, typename Context>
void LabelSmoothGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           float epsilon,
                           phi::DenseTensor* x) {
  auto stream = dev_ctx.stream();
  LabelSmoothMuls<T, Context>(dev_ctx, &out, 1 - epsilon, x);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(label_smooth,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LabelSmoothKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(label_smooth_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LabelSmoothGradKernel,
                          float,
                          phi::dtype::float16) {}
