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

#include <set>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

static phi::DDim GetOutputShape(const std::vector<int64_t> unsqz_dims,
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
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DataType out_dtype,
                  phi::DenseTensor* out) {
  auto dims = axes.GetData();
  dev_ctx.Alloc(out, out->dtype());

  // special case
  if (x.dims().size() == 1 && keep_dim == false) {
    keep_dim = true;
  }

  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  if (x.dims().size() == 0) {
    const auto& cast_runner = NpuOpRunner(
        "Cast",
        {x},
        {*out},
        {{"dst_tytpe", static_cast<int>(ConvertToNpuDtype(out->dtype()))}});
    cast_runner.Run(stream);
    return;
  }

  phi::DenseTensor cast_x;
  phi::DenseTensor cast_out;

  if (out->dtype() == phi::DataType::BOOL) {
    if (x.dtype() == phi::DataType::FLOAT32) {
      cast_x = x;
    } else {
      cast_x.Resize(x.dims());
      dev_ctx.template Alloc<float>(&cast_x);

      const auto& cast_runner =
          NpuOpRunner("Cast", {x}, {cast_x}, {{"dst_tytpe", ACL_FLOAT}});
      cast_runner.Run(stream);
    }

    cast_out.Resize(out->dims());
    dev_ctx.template Alloc<float>(&cast_out);
  } else if (out->dtype() == x.dtype()) {
    cast_x = x;
    cast_out = *out;
  } else {
    phi::DenseTensorMeta meta = {out->dtype(), x.dims()};
    cast_x.set_meta(meta);
    dev_ctx.Alloc(&cast_x, cast_x.dtype());

    const auto& cast_runner = NpuOpRunner(
        "Cast",
        {x},
        {cast_x},
        {{"dst_tytpe", static_cast<int>(ConvertToNpuDtype(out->dtype()))}});
    cast_runner.Run(stream);

    cast_out = *out;
  }

  reduce_all = (reduce_all || axes.size() == 0 || x.dims().size() == 0 ||
                static_cast<int>(axes.size()) == x.dims().size());

  if (reduce_all) {
    std::vector<int> dim_vec;
    for (int i = 0; i < x.dims().size(); i++) {
      dim_vec.push_back(i);
    }

    NpuOpRunner runner;
    runner.SetType("ReduceSum")
        .AddInput(cast_x)
        .AddInput(dev_ctx, std::move(dim_vec))
        .AddOutput(cast_out)
        .AddAttr("keep_dims", keep_dim);
    runner.Run(stream);
  } else {
    NpuOpRunner runner;
    runner.SetType("ReduceSum")
        .AddInput(cast_x)
        .AddInput(dev_ctx, std::move(dims))
        .AddOutput(cast_out)
        .AddAttr("keep_dims", keep_dim);
    runner.Run(stream);
  }

  if (out->dtype() == phi::DataType::BOOL) {
    auto dst_dtype = ConvertToNpuDtype(out->dtype());
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
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  if (dims.size() == 0 || x.dims().size() == 0 ||
      static_cast<int>(dims.size()) == x.dims().size()) {
    reduce_all = true;
  }
  custom_kernel::SumRawKernel<T>(
      dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

template <typename T, typename Context>
void SumGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out_grad,
                   const phi::IntArray& dims_array,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  dev_ctx.Alloc(x_grad, x_grad->dtype());
  auto stream = dev_ctx.stream();

  phi::DenseTensor out_grad_tmp;
  if (x_grad->dtype() == out_grad.dtype()) {
    out_grad_tmp = out_grad;
  } else {
    phi::DenseTensorMeta meta = {x_grad->dtype(), out_grad.dims()};
    out_grad_tmp.set_meta(meta);
    dev_ctx.Alloc(&out_grad_tmp, out_grad_tmp.dtype());

    const auto& cast_runner = NpuOpRunner(
        "Cast",
        {out_grad},
        {out_grad_tmp},
        {{"dst_type", static_cast<int>(ConvertToNpuDtype(x_grad->dtype()))}});
    cast_runner.Run(stream);
  }

  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, out_grad_tmp, true, x_grad);
    return;
  }

  auto keep_dims = keep_dim;

  auto dims = dims_array.GetData();

  // The dims has full dim, set the reduce_all is True
  const auto& input_dim_size = x.dims().size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim || dims.size() == 0);

  if (keep_dims || reduce_all) {
    NpuOpRunner runner;
    runner.SetType("BroadcastTo")
        .AddInput(out_grad_tmp)
        .AddInput(dev_ctx, phi::vectorize(x.dims()))
        .AddOutput(*x_grad);
    runner.Run(stream);
  } else {
    phi::DDim out_dims;
    out_dims = GetOutputShape(dims, out_grad.dims());

    out_grad_tmp.Resize(out_dims);

    NpuOpRunner runner;
    runner.SetType("BroadcastTo")
        .AddInput(out_grad_tmp)
        .AddInput(dev_ctx, phi::vectorize(x.dims()))
        .AddOutput(*x_grad);
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sum_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SumRawKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SumGradKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
