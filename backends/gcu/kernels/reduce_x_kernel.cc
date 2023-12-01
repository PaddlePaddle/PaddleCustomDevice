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

#include "kernels/common_ops/common_ops.h"
#include "kernels/common_ops/reduce_x_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ReduceBaseKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::IntArray& dims,
                      bool keep_dim,
                      bool reduce_all,
                      phi::DenseTensor* out,
                      const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, op_type, reduce_base);

    auto axis = dims.GetData();
    if (dims.size() == 0) {
      axis.assign(x.dims().size(), 0);
      std::iota(axis.begin(), axis.end(), 0);
    }

    auto tmp_x = x;
    auto tmp_out = *out;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
      tmp_out = EmptyTensor(dev_ctx, phi::DataType::INT32, out->dims());
    }

    if (op_type == "reduce_max") {
      reduce_max_compute(dev_ctx, tmp_x, keep_dim, axis, tmp_out);
    } else if (op_type == "reduce_sum") {
      reduce_sum_compute(dev_ctx, tmp_x, keep_dim, axis, tmp_out);
    } else if (op_type == "reduce_prod") {
      reduce_prod_compute(dev_ctx, tmp_x, keep_dim, axis, tmp_out);
    } else if (op_type == "reduce_mean") {
      reduce_mean_compute(dev_ctx, tmp_x, keep_dim, axis, tmp_out);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Aot unsupport reduce type: %s",
                                              op_type.c_str()));
    }

    if (x.dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    }

    PADDLE_GCU_KERNEL_END(op_type, reduce_base);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    auto dim = dims.GetData();
    attrs["dim"] = std::vector<int>(dim.begin(), dim.end());
    attrs["keep_dim"] = keep_dim;
    attrs["reduce_all"] = reduce_all;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
  }
}

template <typename T, typename Context>
void ReduceGradBaseKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& out_grad,
                          const phi::IntArray& dims_array,
                          bool keep_dim,
                          bool reduce_all,
                          phi::DenseTensor* x_grad,
                          const std::string& op_type) {
  dev_ctx.template Alloc<T>(x_grad);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names[GradVarName("Out")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  TensorValueMap outputs;

  output_names[GradVarName("X")] = {"x_grad"};
  outputs[GradVarName("X")] = {x_grad};

  GcuAttributeMap attrs;
  auto dim = dims_array.GetData();
  attrs["dim"] = std::vector<int>(dim.begin(), dim.end());
  attrs["keep_dim"] = keep_dim;
  attrs["reduce_all"] = reduce_all;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  ReduceBaseKernel<T, Context>(dev_ctx,
                               x,
                               dims,
                               keep_dim,
                               ((dims.size() == 0) ? true : false),
                               out,
                               "reduce_max");
}

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& axes,
                bool keep_dim,
                bool reduce_all,
                phi::DenseTensor* out) {
  ReduceBaseKernel<T, Context>(
      dev_ctx, x, axes, keep_dim, reduce_all, out, "reduce_prod");
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  if (UseScatterMemory() && x.dtype() == phi::DataType::BOOL) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "sum_bool", sum_bool);

    dev_ctx.template Alloc(out, out_dtype);
    auto axis = dims.GetData();
    if (dims.size() == 0) {
      axis.assign(x.dims().size(), 0);
      std::iota(axis.begin(), axis.end(), 0);
    }

    auto tmp_x = cast(dev_ctx, x, phi::DataType::INT8);
    auto tmp_out = EmptyTensor(dev_ctx, phi::DataType::INT8, out->dims());

    reduce_sum_compute(dev_ctx, tmp_x, keep_dim, axis, tmp_out);

    // TODO(hongjun): cast to out_dtype
    *out = cast(dev_ctx, tmp_out, phi::DataType::INT64);

    PADDLE_GCU_KERNEL_END("sum_bool", sum_bool);
  } else {
    ReduceBaseKernel<T, Context>(dev_ctx,
                                 x,
                                 dims,
                                 keep_dim,
                                 ((dims.size() == 0) ? true : false),
                                 out,
                                 "reduce_sum");
  }
}

template <typename T, typename Context>
void SumGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out_grad,
                   const phi::IntArray& dims_array,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, out_grad, true, x_grad);
    return;
  }

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "sum_grad", sum_grad);

    std::vector<int64_t> axis;
    int64_t input_rank = x.dims().size();
    if (reduce_all) {
      if (input_rank > 0) {
        axis.assign(input_rank, 0);
        std::iota(axis.begin(), axis.end(), 0);
      }
    } else {
      for (auto value : dims_array.GetData()) {
        if (value >= 0)
          axis.emplace_back(static_cast<int64_t>(value));
        else
          axis.emplace_back(static_cast<int64_t>(value) + input_rank);
      }
    }

    auto tmp_out_grad = out_grad;
    if (out_grad.dtype() == phi::DataType::INT64) {
      tmp_out_grad = cast(dev_ctx, tmp_out_grad, phi::DataType::INT32);
    }

    phi::DenseTensor tmp_x_grad = tmp_out_grad;
    auto output_rank = tmp_out_grad.dims().size();
    if (keep_dim) {
      auto output_shape = phi::vectorize(tmp_out_grad.dims());
      std::vector<int64_t> new_shape;
      size_t iter = 0;
      for (int64_t i = 0; i < output_rank; ++i) {
        if (iter >= axis.size() || i != axis[iter]) {
          new_shape.emplace_back(output_shape[i]);
        } else {
          ++iter;
        }
      }
      tmp_x_grad = reshape(dev_ctx, tmp_x_grad, new_shape);
    }

    std::vector<int64_t> broadcast_dims;
    size_t iter = 0;
    for (int64_t i = 0; i < input_rank; ++i) {
      if (iter >= axis.size() || i != axis[iter]) {
        broadcast_dims.emplace_back(i);
      } else {
        ++iter;
      }
    }
    broadcast_dims.resize(tmp_x_grad.dims().size());
    tmp_x_grad = broadcast_in_dim(
        dev_ctx, tmp_x_grad, phi::vectorize(x.dims()), broadcast_dims);

    if (x_grad->dtype() == phi::DataType::INT64) {
      *x_grad = cast(dev_ctx, tmp_x_grad, phi::DataType::INT64);
    } else {
      *x_grad = tmp_x_grad;
    }

    PADDLE_GCU_KERNEL_END("sum_grad", sum_grad);
  } else {
    dev_ctx.template Alloc<T>(x_grad);
    ReduceGradBaseKernel<T, Context>(dev_ctx,
                                     x,
                                     out_grad,
                                     dims_array,
                                     keep_dim,
                                     reduce_all,
                                     x_grad,
                                     "reduce_sum_grad");
  }
}

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  ReduceBaseKernel<T, Context>(dev_ctx,
                               x,
                               dims,
                               keep_dim,
                               ((dims.size() == 0) ? true : false),
                               out,
                               "reduce_mean");
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& axes,
                    bool keep_dim,
                    bool reduce_all,
                    phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, out_grad, true, x_grad);
    return;
  }

  ReduceGradBaseKernel<T, Context>(dev_ctx,
                                   x,
                                   out_grad,
                                   axes,
                                   keep_dim,
                                   reduce_all,
                                   x_grad,
                                   "reduce_mean_grad");
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  ReduceBaseKernel<T, Context>(dev_ctx,
                               x,
                               dims,
                               keep_dim,
                               ((dims.size() == 0) ? true : false),
                               out,
                               "reduce_min");
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(max,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(prod,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ProdKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SumGradKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(mean,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MeanKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MeanGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(min,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MinKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double) {}
