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

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      phi::DenseTensor* out);

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          int axis,
                          phi::DenseTensor* dx);

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void MultiplyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out);

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& axes,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void MaskedSelectKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& mask,
                        phi::DenseTensor* out);

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
  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(
        phi::errors::External("NPU only support logits_length is_initialized"));
  }
  bool has_labels_length = labels_length.is_initialized();
  if (!has_labels_length) {
    PADDLE_THROW(
        phi::errors::External("NPU only support labels_length is_initialized"));
  }

  int max_sequence_length = logits.dims()[0];
  int num_sequences = logits.dims()[1];
  int sequence_width = logits.dims()[2];

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
  PADDLE_ENFORCE_GE(blank,
                    0,
                    phi::errors::InvalidArgument("Input(blank) should be "
                                                 "equal or greater than zero "
                                                 "but received %d. ",
                                                 blank));
  PADDLE_ENFORCE_LT(blank,
                    sequence_width,
                    phi::errors::InvalidArgument("Input(blank) should be "
                                                 "less than %d "
                                                 "but received %d. ",
                                                 sequence_width,
                                                 blank));

  auto logits_length_dtype = logits_length.get_ptr()->dtype();
  auto labels_length_dtype = labels_length.get_ptr()->dtype();
  PADDLE_ENFORCE_EQ(
      logits_length_dtype == labels_length_dtype,
      true,
      phi::errors::InvalidArgument("The data type of Input(logits_length) and "
                                   "Input(labels_length) should be equal. "));
  PADDLE_ENFORCE_EQ(logits_length_dtype == phi::DataType::INT32 ||
                        logits_length_dtype == phi::DataType::INT64,
                    true,
                    phi::errors::InvalidArgument(
                        "The data type of Input(logits_length) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        phi::DataTypeToString(phi::DataType::INT32),
                        phi::DataTypeToString(phi::DataType::INT64),
                        phi::DataTypeToString(logits_length_dtype)));
  PADDLE_ENFORCE_EQ(labels_length_dtype == phi::DataType::INT32 ||
                        labels_length_dtype == phi::DataType::INT64,
                    true,
                    phi::errors::InvalidArgument(
                        "The data type of Input(labels_length) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        phi::DataTypeToString(phi::DataType::INT32),
                        phi::DataTypeToString(phi::DataType::INT64),
                        phi::DataTypeToString(labels_length_dtype)));

  // Step 1: LogSoftmax
  phi::DenseTensor log_probs;
  log_probs.set_meta(logits.meta());
  log_probs.Resize(logits.dims());
  custom_kernel::LogSoftmaxKernel<T, Context>(dev_ctx, logits, 2, &log_probs);

  std::vector<int64_t> logits_length_vec;
  std::vector<int64_t> labels_length_vec;
  if (logits_length_dtype == phi::DataType::INT32) {
    auto logits_length_vec_ =
        get_new_data_from_tensor<int>(dev_ctx, logits_length.get_ptr());
    auto labels_length_vec_ =
        get_new_data_from_tensor<int>(dev_ctx, labels_length.get_ptr());
    logits_length_vec = std::vector<int64_t>(logits_length_vec_.begin(),
                                             logits_length_vec_.end());
    labels_length_vec = std::vector<int64_t>(labels_length_vec_.begin(),
                                             labels_length_vec_.end());
  } else {
    logits_length_vec =
        get_new_data_from_tensor<int64_t>(dev_ctx, logits_length.get_ptr());
    labels_length_vec =
        get_new_data_from_tensor<int64_t>(dev_ctx, labels_length.get_ptr());
  }

  // Step 2: CtcLoss Forward
  phi::DenseTensor int64_label;
  if (label.dtype() == phi::DataType::INT64) {
    int64_label = label;
  } else {
    phi::DenseTensorMeta int64_label_meta = {phi::DataType::INT64,
                                             label.dims()};
    int64_label.set_meta(int64_label_meta);
    int64_label.Resize(label.dims());
    custom_kernel::CastKernel<int, Context>(
        dev_ctx, label, phi::DataType::INT64, &int64_label);
  }

  phi::DenseTensor non_blank_mask, num_non_blank;
  phi::DenseTensorMeta non_blank_mask_meta = {phi::DataType::BOOL,
                                              int64_label.dims()},
                       num_non_blank_meta = {phi::DataType::INT64,
                                             phi::make_ddim({1})};
  non_blank_mask.set_meta(non_blank_mask_meta);
  num_non_blank.set_meta(num_non_blank_meta);
  dev_ctx.template Alloc<bool>(&non_blank_mask);
  dev_ctx.template Alloc<int64_t>(&num_non_blank);
  phi::Scalar blank_(blank);
  EXEC_NPU_CMD(aclnnNeScalar, dev_ctx, int64_label, blank_, non_blank_mask);
  std::vector<int64_t> reduce_axes(int64_label.dims().size());
  std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
  custom_kernel::SumKernel<bool, Context>(dev_ctx,
                                          non_blank_mask,
                                          phi::IntArray(reduce_axes),
                                          phi::DataType::INT64,
                                          true,
                                          &num_non_blank);
  std::vector<int64_t> num_non_blank_vec(1);
  TensorToVector(dev_ctx, num_non_blank, dev_ctx, &num_non_blank_vec);

  PADDLE_ENFORCE_EQ(
      num_non_blank_vec[0] == std::accumulate(labels_length_vec.begin(),
                                              labels_length_vec.end(),
                                              0),
      true,
      phi::errors::InvalidArgument(
          "The number of non-blank labels should be equal to the sum of "
          "labels_length, "
          "but received non-blank labels: %d, sum of labels_length: %d.",
          num_non_blank_vec[0],
          std::accumulate(
              labels_length_vec.begin(), labels_length_vec.end(), 0)));

  phi::DenseTensor warpctc_label;
  phi::DenseTensorMeta warpctc_label_meta = {
      phi::DataType::INT64, phi::make_ddim({num_non_blank_vec[0]})};
  warpctc_label.set_meta(warpctc_label_meta);
  dev_ctx.template Alloc<int64_t>(&warpctc_label);
  custom_kernel::MaskedSelectKernel<int64_t, Context>(
      dev_ctx, int64_label, non_blank_mask, &warpctc_label);

  int64_t max_label_length =
      *std::max_element(labels_length_vec.begin(), labels_length_vec.end());
  int64_t alpha_tail_size = 2 * max_label_length + 1;
  // Apply for a 32 byte aligned space to avoid address shifting in the OP.
  int64_t alpha_tail_size_align = (alpha_tail_size + 7) / 8 * 8;

  phi::DenseTensor log_alpha;
  phi::DenseTensorMeta log_alpha_meta = {
      logits.dtype(),
      phi::DDim({num_sequences, max_sequence_length, alpha_tail_size_align})};
  log_alpha.set_meta(log_alpha_meta);
  dev_ctx.template Alloc<T>(&log_alpha);

  loss->Resize(phi::make_ddim({num_sequences}));
  dev_ctx.template Alloc<T>(loss);

  bool zero_infinity = false;
  EXEC_NPU_CMD(aclnnCtcLoss,
               dev_ctx,
               log_probs,
               warpctc_label,
               logits_length_vec,
               labels_length_vec,
               blank,
               zero_infinity,
               *loss,
               log_alpha);

  if (warpctcgrad != nullptr) {
    // Step 3: CtcLoss Backward (Compute loss and gradient in one call, as
    // CPU/GPU implementation does)
    phi::DenseTensor ones, log_probs_grad;
    ones.set_meta(loss->meta());
    log_probs_grad.set_meta(log_probs.meta());
    custom_kernel::FullLikeKernel<T, Context>(
        dev_ctx, *loss, phi::Scalar(1.0), loss->dtype(), &ones);
    log_probs_grad.Resize(log_probs.dims());
    dev_ctx.template Alloc<T>(&log_probs_grad);

    EXEC_NPU_CMD(aclnnCtcLossBackward,
                 dev_ctx,
                 ones,
                 log_probs,
                 warpctc_label,
                 logits_length_vec,
                 labels_length_vec,
                 *loss,
                 log_alpha,
                 blank,
                 zero_infinity,
                 log_probs_grad);

    // Step 4: LogSoftmax Backward
    warpctcgrad->Resize(
        phi::make_ddim({max_sequence_length, num_sequences, sequence_width}));
    custom_kernel::LogSoftmaxGradKernel<T, Context>(
        dev_ctx, log_probs, log_probs_grad, 2, warpctcgrad);
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
  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(
        phi::errors::External("NPU only support logits_length is_initialized"));
  }

  custom_kernel::MultiplyRawKernel<T, Context>(
      dev_ctx, warpctcgrad, loss_grad, 1, logits_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    warpctc, npu, ALL_LAYOUT, custom_kernel::WarpctcKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(warpctc_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::WarpctcGradKernel,
                          float,
                          double) {}
