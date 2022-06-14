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
void WhereIndexKernel(const Context& dev_ctx,
                      const phi::DenseTensor& condition,
                      phi::DenseTensor* out) {
  auto dims = condition.dims();
  const int rank = dims.size();

  const aclrtStream& stream = dev_ctx.stream();

  // Run Cast and ReduceSum to get 0 dim of Out
  phi::DenseTensor booled_cond;
  if (condition.dtype() != phi::DataType::BOOL) {
    booled_cond.Resize(dims);
    auto bool_type = ConvertToNpuDtype(phi::DataType::BOOL);
    booled_cond.Resize(dims);
    dev_ctx.template Alloc<bool>(&booled_cond);
    const auto& booled_runner =
        NpuOpRunner("Cast",
                    {condition},
                    {booled_cond},
                    {{"dst_type", static_cast<int32_t>(bool_type)}});
    booled_runner.Run(stream, true);
  } else {
    booled_cond = condition;
  }

  phi::DenseTensor casted_cond;
  auto dst_dtype = ConvertToNpuDtype(phi::DataType::INT64);
  casted_cond.Resize(dims);
  dev_ctx.template Alloc<int64_t>(&casted_cond);
  const auto& cast_runner =
      NpuOpRunner("Cast",
                  {booled_cond},
                  {casted_cond},
                  {{"dst_type", static_cast<int>(dst_dtype)}});
  cast_runner.Run(stream, true);

  phi::DenseTensor sumed_true_num;
  sumed_true_num.Resize({1});
  dev_ctx.template Alloc<int64_t>(&sumed_true_num);
  phi::DenseTensor cond_axes;
  cond_axes.Resize({dims.size()});
  dev_ctx.template Alloc<int>(&cond_axes);
  std::vector<int> axes_vec;
  for (int i = 0; i < dims.size(); ++i) {
    axes_vec.push_back(i);
  }
  custom_kernel::TensorFromVector(dev_ctx, axes_vec, dev_ctx, &cond_axes);
  const auto& sum_runner = NpuOpRunner("ReduceSum",
                                       {casted_cond, cond_axes},
                                       {sumed_true_num},
                                       {{"keep_dims", false}});
  sum_runner.Run(stream, true);

  phi::DenseTensor local_true_num;
  TensorCopy(dev_ctx, sumed_true_num, true, &local_true_num, phi::CPUPlace());
  auto true_num = *local_true_num.data<int64_t>();

  out->Resize(phi::make_ddim({true_num, rank}));
  dev_ctx.template Alloc<int64_t>(out);

  if (true_num == 0) {
    return;
  }

  phi::DenseTensorMeta out_meta = {out->dtype(), out->dims(), out->layout()};
  out->set_meta(out_meta);
  NpuOpRunner runner{"Where", {condition}, {*out}};
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(where_index,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::WhereIndexKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double) {}
