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

#include "paddle/phi/core/utils/data_type.h"

namespace custom_kernel {

template <typename T, typename Context>
struct VisitDataArgNPUMaxFunctor {
  const Context& dev_ctx;
  const phi::DenseTensor& x;
  int64_t axis;
  bool keepdims;
  bool flatten;
  int dtype;
  phi::DenseTensor* out;

  explicit VisitDataArgNPUMaxFunctor(const Context& dev_ctx,
                                     const phi::DenseTensor& x,
                                     int64_t axis,
                                     bool keepdims,
                                     bool flatten,
                                     int dtype,
                                     phi::DenseTensor* out)
      : dev_ctx(dev_ctx),
        x(x),
        axis(axis),
        keepdims(keepdims),
        flatten(flatten),
        dtype(dtype),
        out(out) {}

  template <typename Tout>
  void apply() const {
    LOG(WARNING) << "!!!!!!!!!!!!!! here !!!!!!!!!!!!";
    LOG(WARNING) << "x.dims(): " << x.dims();

    dev_ctx.template Alloc<Tout>(out);
    auto stream = dev_ctx.stream();

    NpuOpRunner runner;
    runner.SetType("ArgMaxV2")
        .AddInput(x)
        .AddInput(dev_ctx, std::vector<int64_t>{axis})
        .AddOutput(*out)
        .AddAttrDataType("dtype", dtype)
        .Run(stream);
  }
};

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int64_t axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  LOG(WARNING) << "!!!!!!!!!!!!!! here !!!!!!!!!!!!";

  LOG(WARNING) << "x.dims(): " << x.dims();

  if (dtype < 0) {
    // if (dtype == phi::DataType::UNDEFINED) {
    LOG(WARNING) << "!!!!!!!!!!!!!! here !!!!!!!!!!!!";
    phi::VisitDataTypeTiny(
        phi::DataType::INT64,
        custom_kernel::VisitDataArgNPUMaxFunctor<T, Context>(
            dev_ctx, x, axis, keepdims, flatten, dtype, out));
    return;
  }
  LOG(WARNING) << "!!!!!!!!!!!!!! here !!!!!!!!!!!!";
  phi::VisitDataTypeTiny(
      // static_cast<phi::DataType>(dtype),
      phi::DataType::INT64,
      // dtype,
      custom_kernel::VisitDataArgNPUMaxFunctor<T, Context>(
          dev_ctx, x, axis, keepdims, flatten, dtype, out));
}

}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(
//     arg_max, ascend, ALL_LAYOUT, custom_kernel::ArgMaxKernel,
//     float,
//     phi::dtype::float16,
//     double) {}
