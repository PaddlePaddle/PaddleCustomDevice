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
void SequenceMaskKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const paddle::optional<phi::DenseTensor>& max_len_tensor,
                        int maxlen,
                        phi::DataType out_dtype,
                        phi::DenseTensor* y) {
  auto stream = dev_ctx.stream();
  if(max_len_tensor){
    auto& max_len = *max_len_tensor.get_ptr();
    phi::DenseTensor local_max_tensor;
    TensorCopy(dev_ctx, max_len, true, &local_max_tensor, phi::CPUPlace());
    maxlen = static_cast<int>(*(local_max_tensor.data<T>()));
  }
  if(maxlen == -1){
    auto x_numel = x.numel();
    if (x_numel == 0) {
      maxlen = 0;
    } else {
      phi::DenseTensor local;
      TensorCopy(dev_ctx, x, true, &local, phi::CPUPlace());
      auto* local_data = local.data<T>();
      maxlen = static_cast<int>(*std::max_element(local_data, local_data + x_numel));
    }
  }
  PADDLE_ENFORCE_GT(
      maxlen,
      0,
      phi::errors::InvalidArgument(
            "Input(MaxLenTensor) value should be greater than 0. But "
            "received Input(MaxLenTensor) value = %d.",
            maxlen));

  // 0. out shape
  auto out_dims = phi::vectorize<int>(x.dims());
  out_dims.push_back(maxlen); 
  y->Resize(phi::make_ddim(out_dims));
  dev_ctx.template Alloc<int64_t>(y);

  std::vector<int> axes;
  axes.push_back(-1);
  NPUAttributeMap attr_input1 = {{"axes", axes}};
  
  // 1. nx = unsqueeze(x, -1)
  phi::DenseTensor nx;
  auto nx_dims = phi::vectorize<int>(x.dims());
  nx_dims.push_back(1);
  nx.Resize(phi::make_ddim(nx_dims));
  dev_ctx.template Alloc<T>(&nx);
  const auto& unsq_runner = NpuOpRunner("UnsqueezeV2", {x}, {nx}, attr_input1);
  unsq_runner.Run(stream);
  dev_ctx.Wait();
  
  // 2. mask = expand(nx)
  phi::DenseTensor mask_x, tn_x;
  mask_x.Resize(phi::make_ddim(out_dims));
  dev_ctx.template Alloc<T>(&mask_x);
  
  // Expand op doesn't support double and int64.
  // we cast double to float32 to support double dtype for now.
  if (nx.dtype() == phi::DataType::FLOAT64 ||
      nx.dtype() == phi::DataType::INT64) {
    auto float_dtype = ConvertToNpuDtype(phi::DataType::FLOAT32);
    
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, nx.dims()};
    tn_x.set_meta(meta);
    dev_ctx.template Alloc<float>(&tn_x);
    dev_ctx.template Alloc<float>(&mask_x);
    const auto& cast_runner1 =
        NpuOpRunner("Cast", {nx}, {tn_x}, {{"dst_type", static_cast<int>(float_dtype)}});
    cast_runner1.Run(stream);
  } else {
    tn_x = nx;
    dev_ctx.template Alloc<T>(&mask_x);
  }
  dev_ctx.Wait();

  NPUAttributeMap attr_input2 = {{"shape", out_dims}};
  const auto& ep_runner = NpuOpRunner("ExpandD", {tn_x}, {mask_x}, attr_input2);
  ep_runner.Run(stream);

  // 3. arange
  phi::DenseTensor range_vec;
  range_vec.Resize({maxlen});
  dev_ctx.template Alloc<T>(&range_vec);

  // Fix refer to
  // https://gitee.com/ascend/modelzoo/issues/I6K3HN?from=project-issue
  NpuOpRunner mk_runner;
  mk_runner.SetType("Range")
      .AddInput(dev_ctx,
                std::move(std::vector<T>{static_cast<int>(0)}),
                false)
      .AddInput(dev_ctx,
                std::move(std::vector<T>{static_cast<int>(maxlen)}),
                false)
      .AddInput(dev_ctx,
                std::move(std::vector<T>{static_cast<int>(1)}),
                false)
      .AddOutput(range_vec);
  mk_runner.Run(stream);

  dev_ctx.Wait();

  // 4. range_mask = expend(range_vec)
  phi::DenseTensor range_mask, t_range_vec;
  range_mask.Resize(phi::make_ddim(out_dims));
  
  if (range_vec.dtype() == phi::DataType::FLOAT64 ||
      range_vec.dtype() == phi::DataType::INT64) {
    auto float_dtype = ConvertToNpuDtype(phi::DataType::FLOAT32);
    
    phi::DenseTensorMeta meta2 = {phi::DataType::FLOAT32, range_vec.dims()};
    t_range_vec.set_meta(meta2);
    dev_ctx.template Alloc<float>(&t_range_vec);
    dev_ctx.template Alloc<float>(&range_mask);
    const auto& cast_runner2 =
        NpuOpRunner("Cast", {range_vec}, {t_range_vec}, {{"dst_type", static_cast<int>(float_dtype)}});
    cast_runner2.Run(stream);
  } else {
    t_range_vec = range_vec;
    dev_ctx.template Alloc<T>(&range_mask);
  }
  
  const auto& ep_runner2 = NpuOpRunner("ExpandD", {t_range_vec}, {range_mask}, attr_input2);
  ep_runner2.Run(stream);
  dev_ctx.Wait();

  // 5. greater_than(mask_x, range_mask)
  phi::DenseTensor bool_out;
  bool_out.Resize(phi::make_ddim(out_dims));
  dev_ctx.template Alloc<bool>(&bool_out);
  const auto& runner = NpuOpRunner("Greater", {mask_x, range_mask}, {bool_out}, {});
  runner.Run(stream);

  // 6. Cast result, bool ---> int
  auto dst_dtype = ConvertToNpuDtype(out_dtype);
  const auto& runner_cast = NpuOpRunner(
      "Cast", {bool_out}, {*y}, {{"dst_type", static_cast<int>(dst_dtype)}});
  runner_cast.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sequence_mask,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SequenceMaskKernel,
                          float,
                          double,
                          int32_t,
                          int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
