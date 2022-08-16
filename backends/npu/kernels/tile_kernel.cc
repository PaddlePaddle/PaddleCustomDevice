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
void TileKernelImpl(const Context& dev_ctx,
                    const phi::DenseTensor& x, 
                    std::vector<int64_t>& repeat_times, 
                    phi::DenseTensor* out) {
  printf("TileKernelImpl");
  auto in_dims = x.dims();
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    PADDLE_ENFORCE_GT(
        repeat_times[i],
        0,
        phi::errors::InvalidArgument(
            "All elements of the input 'repeat_times' for tile op must "
            "be positive integers, but the value received is %d.",
            repeat_times[i]));
  }
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  if (repeat_times.size() < vec_in_dims.size()) {
    int diff = vec_in_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  }
  PADDLE_ENFORCE_EQ(
      repeat_times.size(),
      vec_in_dims.size(),
      phi::errors::InvalidArgument(
          "The rank (%d) of the input 'x' and the rank (%d) of the input "
          "'repeat_times' for tile op must match after promotion.",
          vec_in_dims.size(),
          repeat_times.size()));

  
  phi::DenseTensor* real_time_tensor;
  dev_ctx.template Alloc<int64_t>(real_time_tensor);
  custom_kernel::TensorFromVector<int64_t>(dev_ctx, repeat_times, dev_ctx, real_time_tensor);

  phi::DDim new_in_dims = phi::make_ddim(vec_in_dims);
  phi::DDim out_dims(new_in_dims);

  for (size_t i = 0; i < repeat_times.size(); ++i) {
    out_dims[i] *= repeat_times[i];
  }

  out->Resize(out_dims);
  // out->mutable_data<T>(context.GetPlace());
  dev_ctx.template Alloc<T>(out);

  // std::vector<int> temp(repeat_times.size(), 1);
  // if (repeat_times == temp) {
  //   framework::TensorCopy(*x,
  //                         context.GetPlace(),
  //                         context.template device_context<NPUDeviceContext>(),
  //                         out);
  //   return;
  // }

  auto stream = dev_ctx.stream();
  NpuOpRunner runner;
  runner.SetType("Tile")
      .AddInput(x)
      .AddInput(*real_time_tensor)
      .AddOutput(*out)
      .Run(stream);
}

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const phi::DenseTensor& x, 
                const phi::IntArray& repeat_times, 
                phi::DenseTensor* out 
               ) {
  printf("tilekernel");
  dev_ctx.template Alloc<T>(out);
  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
    rank,
    1,
    phi::errors::InvalidArgument(
        "The rank of the input 'x' for tile op must be a positive "
        "integer, but the value received is %d.",
        rank));  
  PADDLE_ENFORCE_LE(
    rank,
    MAX_RANK_SUPPORTED,
    phi::errors::InvalidArgument(
        "The rank of the input 'x' for tile op "
        "must be less than or equal to %d, but the value received is %d.",
        MAX_RANK_SUPPORTED,
        rank));
  auto repeat_times_arr = repeat_times.GetData();
  int repeat_times_size = repeat_times_arr.size();
  PADDLE_ENFORCE_GE(
    repeat_times_size,
    1,
    phi::errors::InvalidArgument(
        "The number of elements of the input 'repeat_times' for tile "
        "op must be positive, but the value received is %d.",
        repeat_times_size));
  PADDLE_ENFORCE_LE(
    repeat_times_size,
    MAX_RANK_SUPPORTED,
    phi::errors::InvalidArgument(
        "The number of elements of the input 'repeat_times' for tile op "
        "must be less than or equal to %d, but the value received is %d.",
        MAX_RANK_SUPPORTED,
        repeat_times_size));
  rank = std::max(rank, repeat_times_size);
  TileKernelImpl<T, Context>(dev_ctx, x, repeat_times_arr, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tile,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::TileKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
