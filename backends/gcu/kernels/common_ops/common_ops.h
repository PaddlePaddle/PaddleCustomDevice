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

#include <vector>

#include "common/common.h"
#include "common/utils.h"
#include "kernels/common_ops/elementwise_ops.h"
#include "kernels/common_ops/reduce_x_ops.h"
#include "kernels/common_ops/unary_ops.h"
#include "kernels/funcs/gcu_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "paddle/phi/core/dense_tensor.h"

#pragma once
namespace custom_kernel {

phi::DenseTensor ConvertLayout(const phi::CustomContext& dev_ctx,
                               const phi::DenseTensor& tensor,
                               phi::DataLayout layout);

phi::DenseTensor ConvertNCHWToNHWC(const phi::CustomContext& dev_ctx,
                                   const phi::DenseTensor& tensor);

phi::DenseTensor ConvertNHWCToNCHW(const phi::CustomContext& dev_ctx,
                                   const phi::DenseTensor& tensor);

void transpose(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& src_tensor,
               phi::DenseTensor& dst_tensor,  // NOLINT
               const std::vector<int64_t>& permutation);

phi::DenseTensor transpose(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& src_tensor,
                           const std::vector<int64_t>& permutation);

void dot_general_common(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& lhs,
                        const phi::DenseTensor& rhs,
                        phi::DenseTensor& out,  // NOLINT
                        const std::vector<int64_t>& lhs_batch_dimension,
                        const std::vector<int64_t>& rhs_batch_dimension,
                        const std::vector<int64_t>& lhs_contracting_dimension,
                        const std::vector<int64_t>& rhs_contracting_dimension,
                        const double& alpha = 1.0,
                        const double& beta = 0.0);

phi::DenseTensor dot_general_common(
    const phi::CustomContext& dev_ctx,
    const phi::DenseTensor& lhs,
    const phi::DenseTensor& rhs,
    const std::vector<int64_t>& lhs_batch_dimension,
    const std::vector<int64_t>& rhs_batch_dimension,
    const std::vector<int64_t>& lhs_contracting_dimension,
    const std::vector<int64_t>& rhs_contracting_dimension,
    const double& alpha = 1.0,
    const double& beta = 0.0);

void dot_common(const phi::CustomContext& dev_ctx,
                const phi::DenseTensor& lhs,
                const phi::DenseTensor& rhs,
                phi::DenseTensor& out,  // NOLINT
                const double& alpha = 1.0,
                const double& beta = 0.0);

phi::DenseTensor concat(const phi::CustomContext& dev_ctx,
                        const std::vector<phi::DenseTensor>& input_tensors,
                        int64_t axis);  // NOLINT

void concat(const phi::CustomContext& dev_ctx,
            const std::vector<phi::DenseTensor>& input_tensors,
            int64_t axis,
            phi::DenseTensor& output);  // NOLINT

phi::DenseTensor& broadcast(const phi::CustomContext& dev_ctx,
                            const phi::DenseTensor& src,
                            phi::DenseTensor& desc);  // NOLINT

phi::DenseTensor broadcast_in_dim(const phi::CustomContext& dev_ctx,
                                  const phi::DenseTensor& src,
                                  std::vector<int64_t> output_dims,
                                  std::vector<int64_t> broadcast_dimensions);

phi::DenseTensor broadcast_to(const phi::CustomContext& dev_ctx,
                              const phi::DenseTensor& src,
                              std::vector<int64_t> output_dims);

phi::DenseTensor& fill(const phi::CustomContext& dev_ctx,
                       phi::DenseTensor& dims,  // NOLINT
                       const phi::DenseTensor& value);

phi::DenseTensor& slice(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& input,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& ends,
                        const std::vector<int64_t>& steps,
                        phi::DenseTensor& output);  // NOLINT

phi::DenseTensor reverse(const phi::CustomContext& dev_ctx,
                         const phi::DenseTensor& input,
                         const std::vector<int64_t>& reverse_dims);  // NOLINT

phi::DenseTensor& zeros(const phi::CustomContext& dev_ctx,
                        phi::DenseTensor& src);  // NOLINT

phi::DenseTensor& ones(const phi::CustomContext& dev_ctx,
                       phi::DenseTensor& src);  // NOLINT

phi::DenseTensor zeros_like(const phi::CustomContext& dev_ctx,
                            const phi::DenseTensor& src);

phi::DenseTensor ones_like(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& src);

phi::DenseTensor& neg_infs(const phi::CustomContext& dev_ctx,
                           phi::DenseTensor& src);  // NOLINT

phi::DenseTensor& reshape(const phi::CustomContext& dev_ctx,
                          const phi::DenseTensor& src,
                          phi::DenseTensor& dst);  // NOLINT

phi::DenseTensor reshape(const phi::CustomContext& dev_ctx,
                         const phi::DenseTensor& src,
                         const std::vector<int64_t>& output_dims);

phi::DenseTensor& iota(const phi::CustomContext& dev_ctx,
                       phi::DenseTensor& output,  // NOLINT
                       int64_t dim);

phi::DenseTensor select(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& pred,
                        const phi::DenseTensor& on_true,
                        const phi::DenseTensor& on_false);

phi::DenseTensor expand(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& input,
                        const std::vector<int64_t> expand_shape);

phi::DenseTensor gather(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& input);

phi::DenseTensor stack(const phi::CustomContext& dev_ctx,
                       const std::vector<phi::DenseTensor>& inputs,
                       int64_t axis);

void stack(const phi::CustomContext& dev_ctx,
           const std::vector<phi::DenseTensor>& inputs,
           int64_t axis,
           phi::DenseTensor& output);  // NOLINT

phi::DenseTensor softmax_compute(const phi::CustomContext& dev_ctx,
                                 const phi::DenseTensor& input,
                                 int64_t axis);

void softmax_compute(const phi::CustomContext& dev_ctx,
                     const phi::DenseTensor& input,
                     int64_t axis,
                     phi::DenseTensor& output);  // NOLINT

std::vector<phi::DenseTensor> split(const phi::CustomContext& dev_ctx,
                                    const phi::DenseTensor& x,
                                    int axis,
                                    int num,
                                    const std::vector<int64_t>& sections);

void split(const phi::CustomContext& dev_ctx,
           const phi::DenseTensor& x,
           int axis,
           int num,
           std::vector<int64_t> sections,
           std::vector<phi::DenseTensor*> outs);

template <typename T>
phi::DenseTensor CreateScalarTensor(const phi::CustomContext& dev_ctx,
                                    T value) {
  std::vector<T> vec_value(1, value);
  phi::DenseTensor vec_value_tensor;
  vec_value_tensor.Resize(phi::make_ddim({}));
  custom_kernel::TensorFromValue(dev_ctx, value, dev_ctx, &vec_value_tensor);
  dev_ctx.Wait();
  return vec_value_tensor;
}

template <typename T>
phi::DenseTensor full_like(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& src,
                           T value) {
  phi::DenseTensor dst;
  dst.Resize(src.dims());
  dev_ctx.Alloc<T>(&dst);
  auto value_tensor = CreateScalarTensor<T>(dev_ctx, value);

  return fill(dev_ctx, dst, value_tensor);
}

void cast(const phi::CustomContext& dev_ctx,
          const phi::DenseTensor& x,
          phi::DataType dtype,
          phi::DenseTensor* out);

phi::DenseTensor cast(const phi::CustomContext& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DataType dtype);

phi::DenseTensor& one_hot(const phi::CustomContext& dev_ctx,
                          const phi::DenseTensor& x,
                          int64_t axis,
                          int64_t depth,
                          phi::DenseTensor& out);  // NOLINT

phi::DenseTensor one_hot(const phi::CustomContext& dev_ctx,
                         const phi::DenseTensor& x,
                         int64_t axis,
                         int64_t depth);

}  // namespace custom_kernel
