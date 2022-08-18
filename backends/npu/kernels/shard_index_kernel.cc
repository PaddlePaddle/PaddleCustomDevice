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
void SharedIndexKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       int index_num,
                       int nshards,
                       int shard_id,
                       int ignore_value,
                       phi::DenseTensor* out) {
  PADDLE_ENFORCE_GT(
      index_num,
      0,
      phi::errors::InvalidArgument(
          "The value 'index_num' for Op(shard_index) must be greater than 0, "
          "but the value given is %d.",
          index_num));
  PADDLE_ENFORCE_GT(nshards,
                    0,
                    phi::errors::InvalidArgument(
                        "The value 'nshard' for Op(shard_index) must be "
                        "greater than 0, but the value given is %d.",
                        nshards));
  PADDLE_ENFORCE_GE(
      shard_id,
      0,
      phi::errors::InvalidArgument(
          "The value 'shard_id' for Op(shard_index) must be greater or "
          "equal to 0, but the value given is %d.",
          shard_id));
  PADDLE_ENFORCE_LT(
      shard_id,
      nshards,
      phi::errors::InvalidArgument(
          "The value 'shard_id' for Op(shard_index) must be less than "
          "nshards (%d), but the value given is %d.",
          nshards,
          shard_id));

  int shard_size = (index_num + nshards - 1) / nshards;

  out->Resize(x.dims());
  phi::DenseTensor tmp0;
  phi::DenseTensorMeta lod_meta = {x.dtype(), x.dims()};
  tmp0.set_meta(lod_meta);
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor tmp;
  phi::DenseTensorMeta tmp_meta = {x.dtype(), {1}};
  tmp.set_meta(tmp_meta);
  dev_ctx.template Alloc<T>(&tmp);
  FillNpuTensorWithConstant<T>(&tmp, dev_ctx, shard_size);

  phi::DenseTensor condition;
  phi::DenseTensorMeta condition_meta = {phi::DataType::BOOL, x.dims()};
  condition.set_meta(condition_meta);
  dev_ctx.template Alloc<bool>(&condition);

  phi::DenseTensor tmp2;
  phi::DenseTensorMeta tmp2_meta = {x.dtype(), x.dims()};
  tmp2.set_meta(tmp2_meta);
  dev_ctx.template Alloc<T>(&tmp2);

  phi::DenseTensor tmp3;
  phi::DenseTensorMeta tmp3_meta = {x.dtype(), x.dims()};
  tmp3.set_meta(tmp3_meta);
  dev_ctx.template Alloc<T>(&tmp3);

  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Mod", {x, tmp}, {tmp2}, {});
  runner.Run(stream);

  const auto& runner1 = NpuOpRunner("FloorDiv", {x, tmp}, {tmp3}, {});
  runner1.Run(stream);

  FillNpuTensorWithConstant<T>(&tmp, dev_ctx, shard_id);

  const auto& runner2 = NpuOpRunner("Equal", {tmp3, tmp}, {condition}, {});
  runner2.Run(stream);

  phi::DenseTensor tmp4;
  phi::DenseTensorMeta tmp4_meta = {x.dtype(), x.dims()};
  tmp4.set_meta(tmp4_meta);
  dev_ctx.template Alloc<T>(&tmp4);
  FillNpuTensorWithConstant<T>(&tmp4, dev_ctx, ignore_value);

  tmp4.Resize(x.dims());

  const auto& runner3 =
      NpuOpRunner("Select", {condition, tmp2, tmp4}, {*out}, {});
  runner3.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(shard_index,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SharedIndexKernel,
                          int,
                          int64_t) {}
