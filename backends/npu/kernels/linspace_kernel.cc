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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_prepare.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
bool check_tensor_values_in_range(const Context& dev_ctx,
                                  const phi::DenseTensor& x,
                                  phi::DataType dtype = phi::DataType::INT32) {
  if (x.dtype() != phi::DataType::INT64) {
    return true;
  }
  std::vector<int64_t> x_v;
  TensorToVector(dev_ctx, x, dev_ctx, &x_v);
  if (static_cast<int32_t>(x_v[0]) != x_v[0]) {
    return false;
  }
  return true;
}

template <typename T, typename Context>
void AclopLinspaceKernel(const Context& dev_ctx,
                         const phi::DenseTensor& start,
                         const phi::DenseTensor& stop,
                         const phi::DenseTensor& number,
                         phi::DataType dtype,
                         phi::DenseTensor* out) {
  phi::DenseTensor start_n, stop_n, number_n;
  TensorCopy(dev_ctx, start, false, &start_n, phi::CustomPlace());
  TensorCopy(dev_ctx, stop, false, &stop_n, phi::CustomPlace());
  TensorCopy(dev_ctx, number, true, &number_n, phi::CustomPlace());
  std::vector<int32_t> number_v;
  TensorToVector(dev_ctx, number_n, dev_ctx, &number_v);

  PADDLE_ENFORCE_GT(
      number_v[0],
      0,
      phi::errors::InvalidArgument("The num of linspace op should be larger "
                                   "than 0, but received num is %d",
                                   number_v[0]));

  auto stream = dev_ctx.stream();
  phi::DenseTensorMeta out_meta = {dtype, phi::make_ddim({number_v[0]})};
  out->set_meta(out_meta);
  dev_ctx.Alloc(out, out->dtype());

  phi::DenseTensor start_t, stop_t;
  phi::DenseTensorMeta start_meta = {dtype, start.dims()};
  phi::DenseTensorMeta stop_meta = {dtype, stop.dims()};
  start_t.set_meta(start_meta);
  stop_t.set_meta(stop_meta);
  dev_ctx.Alloc(&start_t, start_t.dtype());
  dev_ctx.Alloc(&stop_t, stop_t.dtype());

  custom_kernel::CastKernel<T, Context>(dev_ctx, start_n, dtype, &start_t);

  custom_kernel::CastKernel<T, Context>(dev_ctx, stop_n, dtype, &stop_t);

  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const phi::CustomContext& dev_ctx) {
    const auto& runner = NpuOpRunner(
        "LinSpace", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}, attrs);
    runner.Run(dev_ctx.stream());
  };
  if (dtype == phi::DataType::INT32 || dtype == phi::DataType::INT64) {
    NpuOpRunner::TypeAdapter(
        {start_t, stop_t, number_n},
        {*out},
        {},
        dev_ctx,
        op_func,
        {phi::DataType::FLOAT32, phi::DataType::FLOAT32, phi::DataType::INT32},
        {phi::DataType::FLOAT32});
  } else {
    NpuOpRunner::TypeAdapter(
        {start_t, stop_t, number_n},
        {*out},
        {},
        dev_ctx,
        op_func,
        {start_t.dtype(), stop_t.dtype(), phi::DataType::INT32},
        {out->dtype()});
  }
}

template <typename T, typename Context>
void LinspaceKernel(const Context& dev_ctx,
                    const phi::DenseTensor& start,
                    const phi::DenseTensor& stop,
                    const phi::DenseTensor& number,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnLinspace,
                   (custom_kernel::AclopLinspaceKernel<T, Context>(
                       dev_ctx, start, stop, number, dtype, out)));
  auto cast_dtype =
      dtype == phi::DataType::INT64 ? phi::DataType::INT32 : dtype;
  phi::DenseTensor start_n, stop_n, number_n;
  TensorCopy(dev_ctx, start, false, &start_n, phi::CustomPlace());
  TensorCopy(dev_ctx, stop, false, &stop_n, phi::CustomPlace());
  TensorCopy(dev_ctx, number, true, &number_n, phi::CustomPlace());

  bool start_inrange =
      check_tensor_values_in_range<T, Context>(dev_ctx, start_n);
  bool stop_inrange = check_tensor_values_in_range<T, Context>(dev_ctx, stop_n);
  bool number_inrange =
      check_tensor_values_in_range<T, Context>(dev_ctx, number_n);
  PADDLE_ENFORCE_EQ(
      start_inrange && stop_inrange && number_inrange,
      1,
      phi::errors::InvalidArgument("The size of the input int64 data must be "
                                   "whithin the range of int32."));
  phi::DenseTensor start_t, stop_t;
  phi::DenseTensorMeta start_meta = {cast_dtype, start.dims()};
  phi::DenseTensorMeta stop_meta = {cast_dtype, stop.dims()};
  start_t.set_meta(start_meta);
  stop_t.set_meta(stop_meta);
  dev_ctx.Alloc(&start_t, start_t.dtype());
  dev_ctx.Alloc(&stop_t, stop_t.dtype());

  custom_kernel::CastKernel<T, Context>(dev_ctx, start_n, cast_dtype, &start_t);

  custom_kernel::CastKernel<T, Context>(dev_ctx, stop_n, cast_dtype, &stop_t);

  std::vector<int32_t> number_v;
  TensorToVector(dev_ctx, number_n, dev_ctx, &number_v);
  int64_t num = number_v[0];
  PADDLE_ENFORCE_GT(
      num,
      0,
      phi::errors::InvalidArgument("The num of linspace op should be larger "
                                   "than 0, but received num is %d",
                                   number_v[0]));
  if (dtype == phi::DataType::INT64) {
    std::vector<int32_t> start_v, stop_v;
    TensorToVector(dev_ctx, start_t, dev_ctx, &start_v);
    TensorToVector(dev_ctx, stop_t, dev_ctx, &stop_v);

    phi::Scalar start_scalar = start_v[0];
    phi::Scalar stop_scalar = stop_v[0];

    phi::DenseTensor out_tmp;
    phi::DenseTensorMeta out_meta = {phi::DataType::INT32,
                                     phi::make_ddim({num})};
    out_tmp.set_meta(out_meta);
    dev_ctx.Alloc(&out_tmp, out_tmp.dtype());

    EXEC_NPU_CMD(
        aclnnLinspace, dev_ctx, start_scalar, stop_scalar, num, out_tmp);
    out->Resize(phi::make_ddim({num}));
    custom_kernel::CastKernel<T, Context>(dev_ctx, out_tmp, dtype, out);
  } else {
    std::vector<T> start_v, stop_v;
    TensorToVector(dev_ctx, start_t, dev_ctx, &start_v);
    TensorToVector(dev_ctx, stop_t, dev_ctx, &stop_v);

    phi::Scalar start_scalar = start_v[0];
    phi::Scalar stop_scalar = stop_v[0];

    out->Resize(phi::make_ddim({num}));
    dev_ctx.template Alloc<T>(out);

    EXEC_NPU_CMD(aclnnLinspace, dev_ctx, start_scalar, stop_scalar, num, *out);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(linspace,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LinspaceKernel,
                          float,
                          double,
                          int32_t,
                          int64_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
