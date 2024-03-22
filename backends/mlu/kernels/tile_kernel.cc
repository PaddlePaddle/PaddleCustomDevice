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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {
template <typename T, typename Context>
void TileKernelImpl(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    std::vector<int64_t> repeat_times,
                    int rank,
                    phi::DenseTensor* out) {
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

  bool repeat_one_times = true;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    if (repeat_times[i] != 1) {
      repeat_one_times = false;
    }
  }
  if (rank == 0) {
    TensorCopy(dev_ctx, x, false, out);
    return;
  }
  if (repeat_one_times) {
    TensorCopy(dev_ctx, x, false, out);
  } else {
    phi::DDim new_in_dims = phi::make_ddim(vec_in_dims);
    phi::DDim out_dims(new_in_dims);
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      out_dims[i] *= repeat_times[i];
    }
    out->Resize(out_dims);
    dev_ctx.template Alloc<T>(out);
    MLUCnnlTensorDesc x_desc(x);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::BroadcastTo(
        dev_ctx, x_desc.get(), GetBasePtr(&x), out_desc.get(), GetBasePtr(out));
  }
}

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& repeat_times,
                phi::DenseTensor* out) {
  int rank = static_cast<int>(x.dims().size());
  PADDLE_ENFORCE_GE(rank,
                    0,
                    phi::errors::InvalidArgument(
                        "The rank of the input 'x' for tile op must be a >= 0 "
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
  auto& repeat_times_data = repeat_times.GetData();
  int repeat_times_size = repeat_times_data.size();
  PADDLE_ENFORCE_GE(
      repeat_times_size,
      0,
      phi::errors::InvalidArgument(
          "The number of elements of the input 'repeat_times' for tile "
          "op must be positive, but the value received is %d.",
          repeat_times_size));
  PADDLE_ENFORCE_LE(
      repeat_times_size,
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument(
          "The number of elements of the input 'repeat_times' for tile op "
          "must be >=0, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          repeat_times_size));
  rank = std::max(rank, repeat_times_size);
  TileKernelImpl<T, Context>(dev_ctx, x, repeat_times_data, rank, out);
}

template <typename T, typename Context>
void TileGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& repeat_times,
                    phi::DenseTensor* x_grad) {
  auto x_dims = x.dims();
  auto vec_x_dims = phi::vectorize<int>(x_dims);
  std::vector<int> origin_x_dims = vec_x_dims;
  auto repeat_times_data = repeat_times.GetData();
  if (repeat_times_data.size() < vec_x_dims.size()) {
    int diff = vec_x_dims.size() - repeat_times_data.size();
    repeat_times_data.insert(repeat_times_data.begin(), diff, 1);
  } else {
    int diff = repeat_times_data.size() - vec_x_dims.size();
    vec_x_dims.insert(vec_x_dims.begin(), diff, 1);
  }
  // 1. reshape_dims_vec is the broadcast parameter.
  // 2. reduce_dims_vec is the dimension parameter to compute gradients. For
  //    each dimension expanded, the gradients should be summed to original
  //    size.
  std::vector<int> reshape_dims_vec;
  std::vector<int64_t> reduce_dims_vec;
  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    reduce_dims_vec.push_back(reshape_dims_vec.size());
    reshape_dims_vec.push_back(repeat_times_data[i]);
    reshape_dims_vec.push_back(vec_x_dims[i]);
  }

  int dims = reduce_dims_vec.size();

  bool just_copy = true;
  for (size_t i = 0; i < repeat_times_data.size(); i++) {
    if (repeat_times_data[i] != 1) {
      just_copy = false;
      break;
    }
  }
  // no need reduce, just copy
  if (just_copy) {
    dev_ctx.template Alloc<T>(x_grad);

    TensorCopy(dev_ctx, out_grad, false, x_grad);
    // TensorCopy may change the dims of dx
    x_grad->Resize(x_dims);
  } else {
    PADDLE_ENFORCE_GE(dims,
                      1,
                      phi::errors::InvalidArgument(
                          "Th rank of the input 'Out@GRAD' for tile_grad op "
                          " must be greater than or equal to 1, but "
                          "the value received is %d.",
                          dims));
    PADDLE_ENFORCE_LE(dims,
                      MAX_RANK_SUPPORTED,
                      phi::errors::InvalidArgument(
                          "The rank of the input 'Out@GRAD' for tile_grad op "
                          "must be less than or equal "
                          "to %d, but the value received is %d.",
                          MAX_RANK_SUPPORTED,
                          dims));
    dev_ctx.template Alloc<T>(x_grad);
    phi::DenseTensor dout(out_grad);
    dout.Resize(phi::make_ddim(reshape_dims_vec));

    std::string reduce_name = "reduce_sum";
    MLUReduceOp<T>(dev_ctx,
                   dout,
                   reduce_dims_vec,
                   false, /* keep_dim */
                   false, /* reduce_all */
                   reduce_name,
                   x_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tile,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::TileKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t) {}
PD_REGISTER_PLUGIN_KERNEL(tile_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::TileGradKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t) {}
