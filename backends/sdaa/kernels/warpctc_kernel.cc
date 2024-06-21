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

#include <iostream>
#include <string>
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"
#include "runtime/runtime.h"
#include "tecodnn.h"  // NOLINT

namespace custom_kernel {

template <typename T, typename Context>
void WarpctcKernel(const Context& dev_ctx,
                   const phi::DenseTensor& logits,
                   const phi::DenseTensor& label,
                   const paddle::optional<phi::DenseTensor>& logits_length,
                   const paddle::optional<phi::DenseTensor>& labels_length,
                   int blank,
                   bool norm_by_times,
                   phi::DenseTensor* loss,
                   phi::DenseTensor* warpctcgrad) {
  VLOG(4) << "Call SDAA WarpctcKernel";
  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "SDAA only support logits_length is_initialized"));
  }
  bool has_labels_length = labels_length.is_initialized();
  if (!has_labels_length) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "SDAA only support labels_length is_initialized"));
  }

  int max_sequence_length = logits.dims()[0];
  int num_sequences = logits.dims()[1];
  int sequence_width = logits.dims()[2];
  int max_target_seq_length = label.dims()[1];
  auto logits_length_tensor = logits_length.get();
  auto labels_length_tensor = labels_length.get();
  int logits_length_dims = logits_length_tensor.dims()[0];
  int labels_length_dims = labels_length_tensor.dims()[0];

  PADDLE_ENFORCE_GT(max_sequence_length,
                    0,
                    phi::errors::InvalidArgument(
                        "The first dimension of Input(Logits) should be "
                        "greater than zero "
                        "but received %d. ",
                        max_sequence_length));
  PADDLE_ENFORCE_GT(num_sequences,
                    0,
                    phi::errors::InvalidArgument(
                        "The second dimension of Input(Logits) should be "
                        "greater than zero "
                        "but received %d. ",
                        num_sequences));
  PADDLE_ENFORCE_GT(sequence_width,
                    0,
                    phi::errors::InvalidArgument(
                        "The third dimension of Input(Logits) should be "
                        "greater than zero "
                        "but received %d. ",
                        sequence_width));

  loss->Resize(phi::make_ddim({num_sequences, 1}));
  dev_ctx.template Alloc<T>(loss);

  warpctcgrad->Resize(
      phi::make_ddim({max_sequence_length, num_sequences, sequence_width}));
  dev_ctx.template Alloc<T>(warpctcgrad);

  const T* logits_data = logits.data<T>();
  const int* label_data = label.data<int>();
  T* loss_data = loss->data<T>();
  T* warpctcgrad_data = warpctcgrad->data<T>();

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  if (label.dims()[1] != 1) {
    const auto logits_length_data = logits_length.get_ptr()->data<int64_t>();
    const auto labels_length_data = labels_length.get_ptr()->data<int64_t>();
    std::vector<int> dim_logits = {
        max_sequence_length, num_sequences, sequence_width};
    std::vector<int> dim_label = {num_sequences, max_target_seq_length};
    std::vector<int> dim_logits_length = {logits_length_dims};
    std::vector<int> dim_labels_length = {labels_length_dims};
    std::vector<int> dim_loss = {num_sequences, 1};
    tecodnnTensorDescriptor_t logitsDesc = sdaa_ops::GetTecodnnTensorDesc(
        dim_logits, logits.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t lossDesc = sdaa_ops::GetTecodnnTensorDesc(
        dim_loss, logits.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t labelDesc = sdaa_ops::GetTecodnnTensorDesc(
        dim_label, label.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t gradsDesc = sdaa_ops::GetTecodnnTensorDesc(
        dim_logits, logits.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t logits_lengthDesc =
        sdaa_ops::GetTecodnnTensorDesc(dim_logits_length,
                                       logits_length_tensor.dtype(),
                                       TensorFormat::Undefined);
    tecodnnTensorDescriptor_t labels_lengthDesc =
        sdaa_ops::GetTecodnnTensorDesc(dim_labels_length,
                                       labels_length_tensor.dtype(),
                                       TensorFormat::Undefined);
    TECODNN_CHECK(tecodnnWarpCTCForward(tecodnnHandle,
                                        max_sequence_length,
                                        num_sequences,
                                        sequence_width,
                                        max_target_seq_length,
                                        blank,
                                        logitsDesc,
                                        logits_data,
                                        labelDesc,
                                        label_data,
                                        logits_lengthDesc,
                                        logits_length_data,
                                        labels_lengthDesc,
                                        labels_length_data,
                                        lossDesc,
                                        loss_data,
                                        gradsDesc,
                                        warpctcgrad_data));
    // destroy
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(logitsDesc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(lossDesc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labelDesc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(gradsDesc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(logits_lengthDesc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labels_lengthDesc));
  } else {
    std::vector<int> dim_logits = {
        max_sequence_length, num_sequences, sequence_width};
    tecodnnLossNormalizationMode_t normMode =
        TECODNN_LOSS_NORMALIZATION_SOFTMAX;
    tecodnnNanPropagation_t gradMode = TECODNN_NOT_PROPAGATE_NAN;
    tecodnnCTCLossAlgo_t algo =
        TECODNN_CTC_LOSS_ALGO_DETERMINISTIC;  // set algo, tecodnn is not
                                              // support yet

    tecodnnTensorDescriptor_t logitsDesc = sdaa_ops::GetTecodnnTensorDesc(
        dim_logits, logits.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t gradsDesc = sdaa_ops::GetTecodnnTensorDesc(
        dim_logits, logits.dtype(), TensorFormat::Undefined);
    // declare
    tecodnnCTCLossDescriptor_t ctcLossDesc;

    // create
    TECODNN_CHECK(tecodnnCreateCTCLossDescriptor(&ctcLossDesc));

    // set
    TECODNN_CHECK(tecodnnSetCTCLossDescriptorEx(ctcLossDesc,
                                                TECODNN_DATA_FLOAT,
                                                normMode,
                                                gradMode,
                                                max_sequence_length,
                                                blank));

    // get workspacesize
    size_t workSpaceSizeInBytes = 0;
    TECODNN_CHECK(tecodnnGetCTCLossWorkspaceSize(tecodnnHandle,
                                                 algo,
                                                 ctcLossDesc,
                                                 logitsDesc,
                                                 gradsDesc,
                                                 &workSpaceSizeInBytes));

    phi::DenseTensor workspace;
    T* workspace_data =
        dev_ctx.template Alloc<T>(&workspace, workSpaceSizeInBytes);
    TECODNN_CHECK(tecodnnCTCLoss(tecodnnHandle,
                                 algo,
                                 ctcLossDesc,
                                 logitsDesc,
                                 logits_data,
                                 label_data,
                                 labels_length_tensor.data<int64_t>(),
                                 logits_length_tensor.data<int64_t>(),
                                 loss_data,
                                 gradsDesc,
                                 warpctcgrad_data,
                                 workSpaceSizeInBytes,
                                 workspace_data));

    // destroy
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(logitsDesc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(gradsDesc));
    TECODNN_CHECK(tecodnnDestroyCTCLossDescriptor(ctcLossDesc));
  }
}

template <typename T, typename Context>
void WarpctcGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& logits,
                       const paddle::optional<phi::DenseTensor>& logits_length,
                       const phi::DenseTensor& warpctcgrad,
                       const phi::DenseTensor& loss_grad,
                       int blank,
                       bool norm_by_times,
                       phi::DenseTensor* logits_grad) {
  VLOG(4) << "Call SDAA WarpctcGradKernel";

  dev_ctx.template Alloc<T>(logits_grad);

  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "SDAA only support logits_length is_initialized"));
  }

  int max_sequence_length = warpctcgrad.dims()[0];
  int num_sequences = warpctcgrad.dims()[1];
  int sequence_width = warpctcgrad.dims()[2];
  auto* logits_length_ptr = logits_length.get_ptr();
  auto logits_length_tensor = logits_length.get();
  int logits_length_dims = logits_length_tensor.dims()[0];
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  std::vector<int> dim_logits = {
      max_sequence_length, num_sequences, sequence_width};
  std::vector<int> dim_loss = {num_sequences, 1};
  std::vector<int> dim_logits_length = {logits_length_dims};
  tecodnnTensorDescriptor_t logits_gradDesc = sdaa_ops::GetTecodnnTensorDesc(
      dim_logits, logits.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t loss_gradDesc = sdaa_ops::GetTecodnnTensorDesc(
      dim_loss, logits.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t warpctcgradDesc = sdaa_ops::GetTecodnnTensorDesc(
      dim_logits, logits.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t logits_lengthDesc = sdaa_ops::GetTecodnnTensorDesc(
      dim_logits_length, logits_length_tensor.dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnWarpCTCBackward(tecodnnHandle,
                                       max_sequence_length,
                                       num_sequences,
                                       sequence_width,
                                       norm_by_times,
                                       loss_gradDesc,
                                       loss_grad.data<T>(),
                                       warpctcgradDesc,
                                       warpctcgrad.data<T>(),
                                       logits_lengthDesc,
                                       logits_length_ptr->data<int64_t>(),
                                       logits_gradDesc,
                                       logits_grad->data<T>()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(logits_gradDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(loss_gradDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(warpctcgradDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(logits_lengthDesc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    warpctc, sdaa, ALL_LAYOUT, custom_kernel::WarpctcKernel, float) {}
PD_REGISTER_PLUGIN_KERNEL(
    warpctc_grad, sdaa, ALL_LAYOUT, custom_kernel::WarpctcGradKernel, float) {}
