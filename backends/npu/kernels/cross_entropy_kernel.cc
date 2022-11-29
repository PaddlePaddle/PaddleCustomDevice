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
#include "kernels/funcs/op_command.h"

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

  std::vector<int> axes;
  for (auto i = use_axis; i < logits.dims().size(); ++i) {
    axes.push_back(i);
  }

  experimental::OpCommand("SoftmaxV2")
      .Input(logits,
             experimental::TensorDescMaker("x", logits)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(*softmax,
              experimental::TensorDescMaker("y", *softmax)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Attr("axes", axes)
      .Run(dev_ctx);

  phi::DenseTensor backprop_2d;
  phi::DenseTensorMeta meta = {logits.dtype(), {n, d}};
  backprop_2d.set_meta(meta);
  dev_ctx.template Alloc<T>(&backprop_2d);

  experimental::OpCommand("SparseSoftmaxCrossEntropyWithLogits")
      .Input(logits,
             experimental::TensorDescMaker("features", logits)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims(phi::make_ddim(std::vector<int>({n, d}))))
      .Input(labels,
             experimental::TensorDescMaker("labels", labels)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims(phi::make_ddim(std::vector<int>({n}))))
      .Output(*loss,
              experimental::TensorDescMaker("loss", *loss)
                  .SetDataLayout(phi::DataLayout::ANY)
                  .SetDims(phi::make_ddim(std::vector<int>({n}))))
      .Output(backprop_2d,
              experimental::TensorDescMaker("backprop", backprop_2d)
                  .SetDataLayout(phi::DataLayout::ANY)
                  .SetDims(phi::make_ddim(std::vector<int>({n, d}))))
      .Run(dev_ctx);
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
  phi::DenseTensor on_tensor, off_tensor, depth;
  experimental::OpCommandHelper::ScalarToHostTensor(
      dev_ctx, static_cast<T>(1), &on_tensor);
  experimental::OpCommandHelper::ScalarToHostTensor(
      dev_ctx, static_cast<T>(0), &off_tensor);
  experimental::OpCommandHelper::ScalarToHostTensor(
      dev_ctx, static_cast<int32_t>(softmax.dims()[1]), &depth);
  dev_ctx.template Alloc<T>(logits_grad);

  auto label_dims = phi::vectorize(labels.dims());
  if (label_dims.back() == 1) {
    label_dims.erase(label_dims.cbegin() + label_dims.size() - 1);
  }

  phi::DenseTensor labels_int32, one_hot, softmax_sub_one_hot;
  labels_int32.Resize(phi::make_ddim(label_dims));
  one_hot.Resize(softmax.dims());
  softmax_sub_one_hot.Resize(softmax.dims());
  dev_ctx.template Alloc<int32_t>(&labels_int32);
  dev_ctx.template Alloc<T>(&one_hot);
  dev_ctx.template Alloc<T>(&softmax_sub_one_hot);

  experimental::OpCommand("Cast")
      .Input(labels,
             experimental::TensorDescMaker("x", labels)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(labels_int32,
              experimental::TensorDescMaker("y", labels)
                  .SetDataLayout(
                      phi::DataLayout::ANY))  // the dimention of labels and
                                              // labels_int32 maybe different.
      .Attr("dst_type",
            static_cast<int>(
                experimental::ConvertToNpuDtype(labels_int32.dtype())))
      .Run(dev_ctx);

  experimental::OpCommand("OneHot")
      .Input(labels_int32,
             experimental::TensorDescMaker("x", labels_int32)
                 .SetDataLayout(phi::DataLayout::ANY))
      .ScalarInput(depth,
                   experimental::TensorDescMaker("depth", depth)
                       .SetDataLayout(phi::DataLayout::ANY))
      .ScalarInput(on_tensor,
                   experimental::TensorDescMaker("on_value", on_tensor)
                       .SetDataLayout(phi::DataLayout::ANY))
      .ScalarInput(off_tensor,
                   experimental::TensorDescMaker("off_value", off_tensor)
                       .SetDataLayout(phi::DataLayout::ANY))
      .Output(one_hot,
              experimental::TensorDescMaker("y", one_hot)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Attr("axis", -1)
      .Run(dev_ctx);

  experimental::OpCommand("Sub")
      .Input(softmax,
             experimental::TensorDescMaker("x1", softmax)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(one_hot,
             experimental::TensorDescMaker("x2", one_hot)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(softmax_sub_one_hot,
              experimental::TensorDescMaker("y", softmax_sub_one_hot)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Run(dev_ctx);

  experimental::OpCommand("Mul")
      .Input(loss_grad,
             experimental::TensorDescMaker("x1", loss_grad)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(softmax_sub_one_hot,
             experimental::TensorDescMaker("x2", softmax_sub_one_hot)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(*logits_grad,
              experimental::TensorDescMaker("y", *logits_grad)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                          float,
                          phi::dtype::float16) {}
