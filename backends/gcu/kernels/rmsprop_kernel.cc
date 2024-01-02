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

#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void RmspropDenseKernel(const Context& dev_ctx,
                        const phi::DenseTensor& param,
                        const phi::DenseTensor& mean_square,
                        const phi::DenseTensor& grad,
                        const phi::DenseTensor& moment,
                        const phi::DenseTensor& learning_rate,
                        const paddle::optional<phi::DenseTensor>& mean_grad,
                        const paddle::optional<phi::DenseTensor>& master_param,
                        float epsilon,
                        float decay,
                        float momentum,
                        bool centered,
                        bool multi_precision,
                        phi::DenseTensor* param_out,
                        phi::DenseTensor* moment_out,
                        phi::DenseTensor* mean_square_out,
                        phi::DenseTensor* mean_grad_out,
                        phi::DenseTensor* master_param_outs) {
  PADDLE_ENFORCE_EQ(
      multi_precision,
      false,
      phi::errors::InvalidArgument("Paddle Custom GCU only support "
                                   "multi_precision = false, but got = <%d>",
                                   multi_precision));

  TensorNameMap input_names;
  input_names["Param"] = {"param"};
  input_names["Grad"] = {"grad"};
  input_names["LearningRate"] = {"learning_rate"};
  input_names["Moment"] = {"moment"};
  input_names["MeanSquare"] = {"mean_square"};
  input_names["MeanGrad"] = {"mean_grad"};

  TensorValueMap inputs;
  inputs["Param"] = {const_cast<DenseTensor*>(&param)};
  inputs["Grad"] = {const_cast<DenseTensor*>(&grad)};
  inputs["LearningRate"] = {const_cast<DenseTensor*>(&learning_rate)};
  inputs["Moment"] = {const_cast<DenseTensor*>(&moment)};
  inputs["MeanSquare"] = {const_cast<DenseTensor*>(&mean_square)};
  inputs["MeanGrad"] = {const_cast<DenseTensor*>(&(*mean_grad))};

  phi::DenseTensor param_out_tmp;
  param_out_tmp.set_meta(param_out->meta());
  dev_ctx.template Alloc<T>(&param_out_tmp);

  phi::DenseTensor moment_out_tmp;
  moment_out_tmp.set_meta(moment_out->meta());
  dev_ctx.template Alloc<T>(&moment_out_tmp);

  phi::DenseTensor mean_square_out_tmp;
  mean_square_out_tmp.set_meta(mean_square_out->meta());
  dev_ctx.template Alloc<T>(&mean_square_out_tmp);

  phi::DenseTensor mean_grad_out_tmp;
  mean_grad_out_tmp.set_meta(mean_grad_out->meta());
  dev_ctx.template Alloc<T>(&mean_grad_out_tmp);

  TensorNameMap output_names;
  output_names["ParamOut"] = {"param_out_tmp"};
  output_names["MomentOut"] = {"moment_out_tmp"};
  output_names["MeanSquareOut"] = {"mean_square_out_tmp"};
  output_names["MeanGradOut"] = {"mean_grad_out_tmp"};

  TensorValueMap outputs;
  outputs["ParamOut"] = {&param_out_tmp};
  outputs["MomentOut"] = {&moment_out_tmp};
  outputs["MeanSquareOut"] = {&mean_square_out_tmp};
  outputs["MeanGradOut"] = {&mean_grad_out_tmp};

  GcuAttributeMap attrs;
  attrs["epsilon"] = epsilon;
  attrs["decay"] = decay;
  attrs["momentum"] = momentum;
  attrs["centered"] = centered;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, "rmsprop", dev_ctx);

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(moment_out);
  dev_ctx.template Alloc<T>(mean_square_out);
  dev_ctx.template Alloc<T>(mean_grad_out);
  TensorCopy(dev_ctx, param_out_tmp, false, param_out);
  TensorCopy(dev_ctx, moment_out_tmp, false, moment_out);
  TensorCopy(dev_ctx, mean_square_out_tmp, false, mean_square_out);
  TensorCopy(dev_ctx, mean_grad_out_tmp, false, mean_grad_out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(rmsprop,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::RmspropDenseKernel,
                          float,
                          double) {}
