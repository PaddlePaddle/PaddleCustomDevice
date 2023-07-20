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
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void LinspaceKernel(const Context& dev_ctx,
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

  const auto& cast_runner1 = NpuOpRunner(
      "Cast", {start_n}, {start_t}, {{"dst_type", ConvertToNpuDtype(dtype)}});
  cast_runner1.Run(stream);

  const auto& cast_runner2 = NpuOpRunner(
      "Cast", {stop_n}, {stop_t}, {{"dst_type", ConvertToNpuDtype(dtype)}});
  cast_runner2.Run(stream);

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
