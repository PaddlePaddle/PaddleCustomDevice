/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

static inline int SizeToAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss) {
  PADDLE_ENFORCE_EQ(soft_label,
                    false,
                    phi::errors::Unimplemented(
                        "soft_label=True is not supported in "
                        "the npu kernel of softmax_with_cross_entropy."));

  const int rank = logits.dims().size();
  const int use_axis = axis < 0 ? axis + rank : axis;
  const int n = SizeToAxis(use_axis, logits.dims());
  const int d = SizeFromAxis(use_axis, logits.dims());

  PADDLE_ENFORCE_EQ(
      labels.numel(),
      n,
      phi::errors::Unimplemented(
          "The size of labels should be equal to phi::funcs::SizeToAxis of "
          "logits,"
          "but got size of labels is %d and phi::funcs::SizeToAxis is %d.",
          labels.numel(),
          n));

  dev_ctx.template Alloc<T>(loss);
  dev_ctx.template Alloc<T>(softmax);

  phi::DenseTensor logits_2d(logits), labels_1d(labels), loss_1d(*loss),
      softmax_2d(*softmax);
  logits_2d.Resize({n, d});
  labels_1d.Resize({n});
  loss_1d.Resize({n});
  softmax_2d.Resize({n, d});

  auto stream = dev_ctx.stream();

  std::vector<int> axes;
  for (auto i = use_axis; i < logits.dims().size(); ++i) {
    axes.push_back(i);
  }
  const auto& runner_softmax =
      NpuOpRunner("SoftmaxV2", {logits}, {*softmax}, {{"axes", axes}});
  runner_softmax.Run(stream);

  // SparseSoftmaxCrossEntropyWithLogits
  phi::DenseTensor backprop_2d;
  phi::DenseTensorMeta meta = {logits.dtype(), {n, d}};
  backprop_2d.set_meta(meta);
  dev_ctx.template Alloc<T>(&backprop_2d);

  const auto& runner_s = NpuOpRunner("SparseSoftmaxCrossEntropyWithLogits",
                                     {logits_2d, labels_1d},
                                     {loss_1d, backprop_2d},
                                     {});
  runner_s.Run(stream);
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxGradKernel(const Context& dev_ctx,
                                       const phi::DenseTensor& labels,
                                       const phi::DenseTensor& softmax,
                                       const phi::DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       phi::DenseTensor* logits_grad) {
  int cls_num = softmax.dims()[1];
  auto stream = dev_ctx.stream();

  // cast label from int64/int32 to int32 for OneHotD
  phi::DenseTensor casted_labels;
  if (labels.dtype() != phi::DataType::INT32) {
    phi::DenseTensorMeta casted_labels_meta = {phi::DataType::INT32,
                                               labels.dims()};
    casted_labels.set_meta(casted_labels_meta);
    dev_ctx.template Alloc<int32_t>(&casted_labels);
    auto dst_dtype = ConvertToNpuDtype(phi::DataType::INT32);
    const auto& runner_cast_label =
        NpuOpRunner("Cast",
                    {labels},
                    {casted_labels},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_label.Run(stream);
  } else {
    casted_labels = labels;
  }

  // on and off
  phi::DenseTensor on_tensor, off_tensor;
  phi::DenseTensorMeta on_off_meta = {phi::DataType::INT32, {1}};
  on_tensor.set_meta(on_off_meta);
  off_tensor.set_meta(on_off_meta);
  dev_ctx.template Alloc<int32_t>(&on_tensor);
  dev_ctx.template Alloc<int32_t>(&off_tensor);
  FillNpuTensorWithConstant<int32_t>(&on_tensor, dev_ctx, static_cast<int>(1));
  FillNpuTensorWithConstant<int32_t>(&off_tensor, dev_ctx, static_cast<int>(0));

  // one_hot
  phi::DenseTensor tmp_onehot;
  phi::DenseTensorMeta tmp_onehot_meta = {on_tensor.dtype(), softmax.dims()};
  tmp_onehot.set_meta(tmp_onehot_meta);
  dev_ctx.template Alloc<int32_t>(&tmp_onehot);

  NpuOpRunner runner_onehot;
  runner_onehot.SetType("OneHot")
      .AddInput(casted_labels)
      .AddInput(dev_ctx, std::vector<int32_t>(1, static_cast<int32_t>(cls_num)))
      .AddInput(on_tensor)
      .AddInput(off_tensor)
      .AddOutput(tmp_onehot)
      .AddAttr("axis", -1);
  runner_onehot.Run(stream);

  // cast one_hot from int32 to T
  phi::DenseTensor casted_onehot;
  if (softmax.dtype() != phi::DataType::INT32) {
    phi::DenseTensorMeta onehot_meta = {softmax.dtype(), tmp_onehot.dims()};
    casted_onehot.set_meta(onehot_meta);
    dev_ctx.template Alloc<T>(&casted_onehot);
    auto dst_dtype = ConvertToNpuDtype(softmax.dtype());
    const auto& runner_cast_onehot =
        NpuOpRunner("Cast",
                    {tmp_onehot},
                    {casted_onehot},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_onehot.Run(stream);
  } else {
    casted_onehot = tmp_onehot;
  }

  // sub
  phi::DenseTensor tmp_sub;
  phi::DenseTensorMeta tmp_sub_meta = {softmax.dtype(), softmax.dims()};
  tmp_sub.set_meta(tmp_sub_meta);
  dev_ctx.template Alloc<T>(&tmp_sub);

  const auto& runner_sub =
      NpuOpRunner("Sub", {softmax, casted_onehot}, {tmp_sub}, {});
  runner_sub.Run(stream);

  // mul
  dev_ctx.template Alloc<T>(logits_grad);
  const auto& runner_mul =
      NpuOpRunner("Mul", {loss_grad, tmp_sub}, {*logits_grad}, {});
  runner_mul.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                          float,
                          phi::dtype::float16) {}
