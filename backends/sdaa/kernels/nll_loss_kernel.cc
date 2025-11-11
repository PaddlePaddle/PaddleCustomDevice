// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& labels,
                      const paddle::optional<phi::DenseTensor>& weight,
                      int64_t ignore_index,
                      const std::string& reduction,
                      phi::DenseTensor* out,
                      phi::DenseTensor* total_weight) {
  VLOG(4) << "Call SDAA NllLossRawKernel";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> labels_dims = phi::vectorize<int>(labels.dims());

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<float>(total_weight);

  int batch_size = x_dims[0];
  int n_classes = x_dims[1];

  phi::DenseTensor weight_temp;
  if (weight.get_ptr() == nullptr) {
    std::vector<int> temp = {n_classes};
    phi::DDim weight_dims = phi::make_ddim(std::move(temp));
    weight_temp.Resize(weight_dims);
    dev_ctx.template Alloc<T>(&weight_temp);
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(1), x.dtype(), &weight_temp);
  } else {
    weight_temp = weight.get();
  }

  if (x_dims.size() == 2) {
    std::vector<int> out_dims(2, 1);

    tecodnnLossReductionMode_t reduction_mode;
    // `tecodnnNllLossForward` reduction mode is legacy
    if (reduction == "mean") {
      reduction_mode = TECODNN_LOSS_REDUCTION_MEAN;
    } else if (reduction == "none") {
      reduction_mode = TECODNN_LOSS_REDUCTION_NONE;
      out_dims[0] = batch_size;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "reduction mode only support 'none', 'mean' on %s for x_dims of 2, "
          "but got %s instead",
          dev_ctx.GetPlace(),
          reduction));
    }

    // tecodnn accept labels dims and output dims is [..., 1]
    labels_dims.emplace_back(1);

    std::vector<int> weight_dims = phi::vectorize<int>(weight_temp.dims());
    phi::DenseTensor labels_cast;
    if (labels.dtype() != phi::DataType::INT32) {
      // due to tecodnn is only support labels of INT32 -> do cast
      labels_cast.Resize(labels.dims());
      phi::DenseTensorMeta labels_meta = {phi::DataType::INT32, labels.dims()};
      labels_cast.set_meta(labels_meta);
      dev_ctx.template Alloc<int32_t>(&labels_cast);
      sdaa_ops::doCastTensor(dev_ctx, labels, &labels_cast);
    } else {
      labels_cast = labels;
    }

    tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

    tecodnnTensorDescriptor_t x_desc = sdaa_ops::GetTecodnnTensorDesc(
        x_dims, x.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t labels_desc = sdaa_ops::GetTecodnnTensorDesc(
        labels_dims, labels_cast.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t weight_desc = sdaa_ops::GetTecodnnTensorDesc(
        weight_dims, weight_temp.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t out_desc = sdaa_ops::GetTecodnnTensorDesc(
        out_dims, out->dtype(), TensorFormat::Undefined);

    TECODNN_CHECK(tecodnnNLLLossForward(handle,
                                        ignore_index,
                                        reduction_mode,
                                        x_desc,
                                        x.data(),
                                        labels_desc,
                                        labels_cast.data(),
                                        weight_desc,
                                        weight_temp.data(),
                                        out_desc,
                                        out->data(),
                                        total_weight->data()));

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labels_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(weight_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
  } else if (x_dims.size() == 4) {
    const phi::DDim temp_out_dim = {batch_size, x_dims[2], x_dims[3]};  // NHW
    std::vector<int> out_dims = phi::vectorize<int>(temp_out_dim);

    tecodnnLossReductionMode_t reduction_mode;
    if (reduction == "none") {
      reduction_mode = TECODNN_LOSS_REDUCTION_NONE;
    } else if (reduction == "mean") {
      reduction_mode = TECODNN_LOSS_REDUCTION_MEAN;
    } else if (reduction == "sum") {
      reduction_mode = TECODNN_LOSS_REDUCTION_SUM;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "reduction mode only support 'none', 'mean' and 'sum' on %s for "
          "x_dims of 4, but got %s instead",
          dev_ctx.GetPlace(),
          reduction));
    }

    std::vector<int> weight_dims = phi::vectorize<int>(weight_temp.dims());
    std::vector<int> total_weight_dims =
        phi::vectorize<int>(total_weight->dims());

    tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

    tecodnnTensorDescriptor_t x_desc = sdaa_ops::GetTecodnnTensorDesc(
        x_dims, x.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t labels_desc = sdaa_ops::GetTecodnnTensorDesc(
        labels_dims, labels.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t weight_desc = sdaa_ops::GetTecodnnTensorDesc(
        weight_dims, weight_temp.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t out_desc = sdaa_ops::GetTecodnnTensorDesc(
        out_dims, out->dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t total_weight_desc =
        sdaa_ops::GetTecodnnTensorDesc(
            total_weight_dims, total_weight->dtype(), TensorFormat::Undefined);

    if (reduction == "none") {
      TECODNN_CHECK(tecodnnNLLLoss2dForward(handle,
                                            ignore_index,
                                            reduction_mode,
                                            x_desc,
                                            x.data(),
                                            labels_desc,
                                            labels.data(),
                                            weight_desc,
                                            weight_temp.data(),
                                            out_desc,
                                            out->data(),
                                            total_weight_desc,
                                            total_weight->data()));
    } else {
      phi::DenseTensor out_temp;
      out_temp.Resize(temp_out_dim);
      dev_ctx.template Alloc<T>(&out_temp);
      TECODNN_CHECK(tecodnnNLLLoss2dForward(handle,
                                            ignore_index,
                                            reduction_mode,
                                            x_desc,
                                            x.data(),
                                            labels_desc,
                                            labels.data(),
                                            weight_desc,
                                            weight_temp.data(),
                                            out_desc,
                                            out_temp.data(),
                                            total_weight_desc,
                                            total_weight->data()));

      const std::vector<int> axes = {0, 1, 2};
      const std::vector<int> starts = {0, 0, 0};
      const std::vector<int> ends = {1, 1, 1};
      const std::vector<int> strides(axes.size(), 1);
      const std::vector<int64_t> decreased_dims = {1, 2};
      sdaa_ops::doSliceTensor(
          dev_ctx, out_temp, axes, starts, ends, strides, decreased_dims, out);
    }

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labels_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(weight_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(total_weight_desc));
  }

  if (weight.get_ptr() == nullptr && reduction == "none") {
    // if weight is none && reduction is none -> total weight is 0
    sdaa_ops::doFillTensor<float>(
        dev_ctx, static_cast<float>(0), total_weight->dtype(), total_weight);
  }
}

template <typename T, typename Context>
void NllLossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const paddle::optional<phi::DenseTensor>& weight,
                       const phi::DenseTensor& total_weight,
                       const phi::DenseTensor& dout,
                       int64_t ignore_index,
                       const std::string& reduction,
                       phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA NllLossGradKernel";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> labels_dims = phi::vectorize<int>(labels.dims());
  std::vector<int> dx_dims = phi::vectorize<int>(dx->dims());

  dev_ctx.template Alloc<T>(dx);
  int batch_size = x_dims[0];
  int n_classes = x_dims[1];

  phi::DenseTensor weight_temp;
  if (weight.get_ptr() == nullptr) {
    std::vector<int> temp = {n_classes};
    phi::DDim weight_dims = phi::make_ddim(std::move(temp));
    weight_temp.Resize(weight_dims);
    dev_ctx.template Alloc<T>(&weight_temp);
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(1), x.dtype(), &weight_temp);
  } else {
    weight_temp = weight.get();
  }

  if (x_dims.size() == 2) {
    std::vector<int> dout_dims(2, 1);

    tecodnnLossReductionMode_t reduction_mode;
    // `tecodnnNllLossForward` reduction mode is legacy
    if (reduction == "mean") {
      reduction_mode = TECODNN_LOSS_REDUCTION_MEAN;
    } else if (reduction == "none") {
      reduction_mode = TECODNN_LOSS_REDUCTION_NONE;
      dout_dims[0] = batch_size;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "reduction mode only support 'none', 'mean' on %s for x_dims of 2, "
          "but got %s instead",
          dev_ctx.GetPlace(),
          reduction));
    }

    // tecodnn accept labels dims and dout dims is [..., 1]
    labels_dims.emplace_back(1);

    std::vector<int> weight_dims = phi::vectorize<int>(weight_temp.dims());
    phi::DenseTensor labels_cast;
    if (labels.dtype() != phi::DataType::INT32) {
      // due to tecodnn is only support labels of INT32 -> do cast
      labels_cast.Resize(labels.dims());
      phi::DenseTensorMeta labels_meta = {phi::DataType::INT32, labels.dims()};
      labels_cast.set_meta(labels_meta);
      dev_ctx.template Alloc<int32_t>(&labels_cast);
      sdaa_ops::doCastTensor(dev_ctx, labels, &labels_cast);
    } else {
      labels_cast = labels;
    }

    tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

    tecodnnTensorDescriptor_t labels_desc = sdaa_ops::GetTecodnnTensorDesc(
        labels_dims, labels_cast.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t weight_desc = sdaa_ops::GetTecodnnTensorDesc(
        weight_dims, weight_temp.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t dout_desc = sdaa_ops::GetTecodnnTensorDesc(
        dout_dims, dout.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t dx_desc = sdaa_ops::GetTecodnnTensorDesc(
        dx_dims, dx->dtype(), TensorFormat::Undefined);

    TECODNN_CHECK(tecodnnNLLLossBackward(handle,
                                         ignore_index,
                                         reduction_mode,
                                         labels_desc,
                                         labels_cast.data(),
                                         weight_desc,
                                         weight_temp.data(),
                                         dout_desc,
                                         dout.data(),
                                         total_weight.data(),
                                         dx_desc,
                                         dx->data()));

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labels_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(weight_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dout_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dx_desc));
  } else if (x_dims.size() == 4) {
    tecodnnLossReductionMode_t reduction_mode;
    if (reduction == "none") {
      reduction_mode = TECODNN_LOSS_REDUCTION_NONE;
    } else if (reduction == "mean") {
      reduction_mode = TECODNN_LOSS_REDUCTION_MEAN;
    } else if (reduction == "sum") {
      reduction_mode = TECODNN_LOSS_REDUCTION_SUM;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "reduction mode only support 'none', 'mean' and 'sum' on %s for "
          "x_dims of 4, but got %s instead",
          dev_ctx.GetPlace(),
          reduction));
    }

    std::vector<int> weight_dims = phi::vectorize<int>(weight_temp.dims());
    std::vector<int> total_weight_dims =
        phi::vectorize<int>(total_weight.dims());

    tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

    tecodnnTensorDescriptor_t x_desc = sdaa_ops::GetTecodnnTensorDesc(
        x_dims, x.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t labels_desc = sdaa_ops::GetTecodnnTensorDesc(
        labels_dims, labels.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t weight_desc = sdaa_ops::GetTecodnnTensorDesc(
        weight_dims, weight_temp.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t dout_desc = sdaa_ops::GetTecodnnTensorDesc(
        labels_dims,
        dout.dtype(),
        TensorFormat::Undefined);  // dout dims is NHW same as labels
    tecodnnTensorDescriptor_t total_weight_desc =
        sdaa_ops::GetTecodnnTensorDesc(
            total_weight_dims, total_weight.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t dx_desc = sdaa_ops::GetTecodnnTensorDesc(
        dx_dims, dx->dtype(), TensorFormat::Undefined);

    TECODNN_CHECK(tecodnnNLLLoss2dBackward(handle,
                                           ignore_index,
                                           reduction_mode,
                                           x_desc,
                                           x.data(),
                                           labels_desc,
                                           labels.data(),
                                           weight_desc,
                                           weight_temp.data(),
                                           dout_desc,
                                           dout.data(),
                                           total_weight_desc,
                                           total_weight.data(),
                                           dx_desc,
                                           dx->data()));

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labels_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(weight_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dout_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(total_weight_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dx_desc));
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(nll_loss,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NllLossRawKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  }
}

PD_REGISTER_PLUGIN_KERNEL(nll_loss_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NllLossGradKernel,
                          float,
                          phi::dtype::float16) {}
