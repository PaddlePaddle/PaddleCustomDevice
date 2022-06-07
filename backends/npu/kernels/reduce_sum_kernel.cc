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

static phi::DDim GetOutputShape(const std::vector<long int> unsqz_dims,
                                const phi::DDim& in_dims) {
  int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
  int cur_output_size = in_dims.size();
  std::vector<int64_t> output_shape(output_size, 0);

  // Validity Check: rank range.
  PADDLE_ENFORCE_LE(
      output_size,
      6,
      phi::errors::InvalidArgument("The output "
                                   "tensor's rank should be less than 6."));

  for (int axis : unsqz_dims) {
    int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
    // Vaildity Check: the axis bound
    PADDLE_ENFORCE_GE(
        cur,
        0,
        phi::errors::InvalidArgument("The insert dimension value should "
                                     "not be less than 0"));
    PADDLE_ENFORCE_LE(cur,
                      cur_output_size,
                      phi::errors::InvalidArgument(
                          "The insert dimension value shoule not be larger "
                          "than the dimension size of input tensor"));
    // Move old axis, and insert new axis
    for (int i = cur_output_size; i >= cur; --i) {
      if (output_shape[i] == 1) {
        // Move axis
        output_shape[i + 1] = 1;
        output_shape[i] = 0;
      }
    }
    output_shape[cur] = 1;
    // Add the output size.
    cur_output_size++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
    if (output_shape[out_idx] == 0) {
      output_shape[out_idx] = in_dims[in_idx++];
    }
  }

  return phi::make_ddim(output_shape);
}

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  auto dims = axes;
  dev_ctx.template Alloc<T>(out);

  // special case
  if (x.dims().size() == 1 && keep_dim == false) {
    keep_dim = true;
  }

  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  phi::DenseTensor cast_x;
  phi::DenseTensor cast_out;
  // NOTE: ReduceSumD only supports fp32 and fp16
  if (x.dtype() != phi::DenseTensorMeta::DataType::FLOAT32 &&
      x.dtype() != phi::DenseTensorMeta::DataType::FLOAT16) {
    cast_x.Resize(x.dims());
    dev_ctx.template Alloc<T>(&cast_x);
    cast_out.Resize(out->dims());
    dev_ctx.template Alloc<T>(&cast_out);

    const auto& runner_cast = NpuOpRunner(
        "Cast", {x}, {cast_x}, {{"dst_type", static_cast<int>(ACL_FLOAT)}});
    runner_cast.Run(stream);
  } else {
    cast_x = x;
    cast_out = *out;
  }

  if (reduce_all) {
    std::vector<int> dim_vec;
    for (int i = 0; i < x.dims().size(); i++) {
      dim_vec.push_back(i);
    }

    const auto& runner =
        NpuOpRunner("ReduceSumD",
                    {cast_x},
                    {cast_out},
                    {{"axes", dim_vec}, {"keep_dims", keep_dim}});
    runner.Run(stream);

  } else {
    const auto& runner = NpuOpRunner("ReduceSumD",
                                     {cast_x},
                                     {cast_out},
                                     {{"axes", dims}, {"keep_dims", keep_dim}});
    runner.Run(stream);
  }

  if (x.dtype() != phi::DenseTensorMeta::DataType::FLOAT32 &&
      x.dtype() != phi::DenseTensorMeta::DataType::FLOAT16) {
    auto dst_dtype = ConvertToNpuDtype(out_dtype);
    const auto& runner_cast =
        NpuOpRunner("Cast",
                    {cast_out},
                    {*out},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast.Run(stream);
  }
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  custom_kernel::SumRawKernel<T>(
      dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

template <typename T, typename Context>
void SumGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out_grad,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  auto keep_dims = keep_dim;

  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();

  if (keep_dims || reduce_all) {
    const auto& runner = NpuOpRunner("BroadcastToD",
                                     {out_grad},
                                     {*x_grad},
                                     {{"shape", phi::vectorize(x.dims())}});
    runner.Run(stream);
  } else {
    phi::DDim out_dims;
    out_dims = GetOutputShape(dims, out_grad.dims());

    phi::DenseTensor out_grad_tmp;
    phi::DenseTensorMeta out_grad_tmp_meta = {out_grad.dtype(), out_dims};
    out_grad_tmp.set_meta(out_grad_tmp_meta);
    dev_ctx.template Alloc<T>(&out_grad_tmp);
    TensorCopy(dev_ctx, out_grad, false, &out_grad_tmp);
    out_grad_tmp.Resize(out_dims);

    const auto& runner = NpuOpRunner("BroadcastToD",
                                     {out_grad_tmp},
                                     {*x_grad},
                                     {{"shape", phi::vectorize(x.dims())}});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sum_raw,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SumRawKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SumGradKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}
