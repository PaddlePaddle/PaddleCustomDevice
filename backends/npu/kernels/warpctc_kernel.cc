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
static void FullLike(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& val,
                     phi::DataType dtype,
                     phi::DenseTensor* out) {
  auto value = val.to<double>();
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<
          std::is_same<T, phi::dtype::float16>::value ||
              std::is_same<T, phi::dtype::bfloat16>::value,
          float,
          T>::type>::type;

  auto common_type_value = static_cast<CommonType>(value);

  // Check whether the filled value is valid
  bool is_out_range = true;
  if (std::isinf(value) || std::isnan(value)) {
    is_out_range = false;
  }

  if ((common_type_value >=
       static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
      (common_type_value <=
       static_cast<CommonType>(std::numeric_limits<T>::max()))) {
    is_out_range = false;
  }

  PADDLE_ENFORCE_EQ(
      is_out_range,
      false,
      phi::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          static_cast<float>(value)));

  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnInplaceFillScalar, dev_ctx, *out, val);
}

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
void SliceRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& axes_t,
                    const phi::IntArray& starts_array,
                    const phi::IntArray& ends_array,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& axes,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out);

template <typename T, typename Context>
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out);

template <typename T, typename Context>
void MaskedSelectKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& mask,
                        phi::DenseTensor* out);

template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out);

template <typename Context>
void UnpadLabel(const Context& dev_ctx,
                const phi::DenseTensor& label,
                int blank,
                const std::vector<int64_t>& label_lengths_vec,
                phi::DenseTensor* warpctc_label) {
  phi::DenseTensor non_blank_mask, num_non_blank;
  phi::DenseTensorMeta non_blank_mask_meta = {phi::DataType::BOOL,
                                              label.dims()},
                       num_non_blank_meta = {phi::DataType::INT64,
                                             phi::make_ddim({1})};
  non_blank_mask.set_meta(non_blank_mask_meta);
  num_non_blank.set_meta(num_non_blank_meta);
  dev_ctx.template Alloc<bool>(&non_blank_mask);
  phi::Scalar blank_(blank);
  EXEC_NPU_CMD(aclnnNeScalar, dev_ctx, label, blank_, non_blank_mask);
  std::vector<int64_t> reduce_axes(label.dims().size());
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
      num_non_blank_vec[0] == std::accumulate(label_lengths_vec.begin(),
                                              label_lengths_vec.end(),
                                              0),
      true,
      phi::errors::InvalidArgument(
          "The number of non-blank labels should be equal to the sum of "
          "label_lengths, "
          "but received non-blank labels: %d, sum of label_lengths: %d.",
          num_non_blank_vec[0],
          std::accumulate(
              label_lengths_vec.begin(), label_lengths_vec.end(), 0)));

  phi::DenseTensorMeta warpctc_label_meta = {
      phi::DataType::INT64, phi::make_ddim({num_non_blank_vec[0]})};
  warpctc_label->set_meta(warpctc_label_meta);
  custom_kernel::MaskedSelectKernel<int64_t, Context>(
      dev_ctx, label, non_blank_mask, warpctc_label);
}

template <typename Context>
void CountLabelRepeats(const Context& dev_ctx,
                       const phi::DenseTensor& label,
                       phi::DenseTensor* repeats) {
  phi::DenseTensor label_0, label_1;
  auto slice_label_dims =
      phi::make_ddim({label.dims()[0], label.dims()[1] - 1});
  phi::DenseTensorMeta slice_label_meta = {phi::DataType::INT64,
                                           slice_label_dims};
  label_0.set_meta(slice_label_meta);
  label_1.set_meta(slice_label_meta);
  label_0.Resize(slice_label_dims);
  label_1.Resize(slice_label_dims);
  custom_kernel::SliceRawKernel<int64_t, Context>(
      dev_ctx, label, {1}, {0}, {label.dims()[1] - 1}, {1, 1}, {}, &label_0);
  custom_kernel::SliceRawKernel<int64_t, Context>(
      dev_ctx, label, {1}, {1}, {label.dims()[1]}, {1, 1}, {}, &label_1);

  phi::DenseTensor repeat_mask;
  phi::DenseTensorMeta repeat_mask_meta = {phi::DataType::BOOL, label_0.dims()};
  repeat_mask.set_meta(repeat_mask_meta);
  custom_kernel::EqualKernel<int64_t, Context>(
      dev_ctx, label_0, label_1, &repeat_mask);

  custom_kernel::SumKernel<bool, Context>(dev_ctx,
                                          repeat_mask,
                                          phi::IntArray({1}),
                                          phi::DataType::INT64,
                                          false,
                                          repeats);
}

template <typename T, typename Context>
void GenZeroInfMask(const Context& dev_ctx,
                    const phi::DenseTensor& label,
                    const phi::DenseTensor& label_lengths,
                    const phi::DenseTensor& logits_lengths,
                    phi::DenseTensor* zero_inf_mask) {
  phi::DenseTensor repeats;
  phi::DenseTensorMeta repeats_meta = {phi::DataType::INT64,
                                       label_lengths.dims()};
  repeats.set_meta(repeats_meta);
  custom_kernel::CountLabelRepeats<Context>(dev_ctx, label, &repeats);

  phi::DenseTensor label_dim_tensor, padded_lengths;
  label_dim_tensor.set_meta(label_lengths.meta());
  padded_lengths.set_meta(label_lengths.meta());
  FillNpuTensorWithConstant<int64_t>(
      &label_dim_tensor, dev_ctx, static_cast<int64_t>(label.dims()[1]));
  custom_kernel::SubtractKernel<T, Context>(
      dev_ctx, label_dim_tensor, label_lengths, &padded_lengths);

  phi::DenseTensor ones, padded_lengths_, padded_lengths_clip, total_length,
      total_length_temp;
  ones.set_meta(label_lengths.meta());
  padded_lengths_.set_meta(label_lengths.meta());
  padded_lengths_clip.set_meta(label_lengths.meta());
  total_length_temp.set_meta(label_lengths.meta());
  total_length.set_meta(label_lengths.meta());
  custom_kernel::FullLike<T, Context>(
      dev_ctx, label_lengths, phi::Scalar(1.0), label_lengths.dtype(), &ones);
  custom_kernel::SubtractKernel<T, Context>(
      dev_ctx, padded_lengths, ones, &padded_lengths_);
  dev_ctx.template Alloc<T>(&padded_lengths_clip);
  phi::Scalar min_thresh(0.0);
  EXEC_NPU_CMD(
      aclnnClampMin, dev_ctx, padded_lengths_, min_thresh, padded_lengths_clip);

  custom_kernel::AddKernel<T, Context>(
      dev_ctx, repeats, label_lengths, &total_length_temp);
  custom_kernel::SubtractKernel<T, Context>(
      dev_ctx, total_length_temp, padded_lengths_clip, &total_length);
  custom_kernel::LessEqualKernel<T, Context>(
      dev_ctx, total_length, logits_lengths, zero_inf_mask);
}

template <typename T, typename Context>
void WarpctcKernel(const Context& dev_ctx,
                   const phi::DenseTensor& logits,
                   const phi::DenseTensor& label,
                   const paddle::optional<phi::DenseTensor>& logits_lengths,
                   const paddle::optional<phi::DenseTensor>& label_lengths,
                   int blank,
                   bool norm_by_times,
                   phi::DenseTensor* loss,
                   phi::DenseTensor* warpctcgrad) {
  bool has_logits_lengths = logits_lengths.is_initialized();
  if (!has_logits_lengths) {
    PADDLE_THROW(phi::errors::External(
        "NPU only support logits_lengths is_initialized"));
  }
  bool has_label_lengths = label_lengths.is_initialized();
  if (!has_label_lengths) {
    PADDLE_THROW(
        phi::errors::External("NPU only support label_lengths is_initialized"));
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

  auto logits_lengths_dtype = logits_lengths.get_ptr()->dtype();
  auto label_lengths_dtype = label_lengths.get_ptr()->dtype();
  PADDLE_ENFORCE_EQ(
      logits_lengths_dtype == label_lengths_dtype,
      true,
      phi::errors::InvalidArgument("The data type of Input(logits_lengths) and "
                                   "Input(label_lengths) should be equal. "));
  PADDLE_ENFORCE_EQ(logits_lengths_dtype == phi::DataType::INT32 ||
                        logits_lengths_dtype == phi::DataType::INT64,
                    true,
                    phi::errors::InvalidArgument(
                        "The data type of Input(logits_lengths) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        phi::DataTypeToString(phi::DataType::INT32),
                        phi::DataTypeToString(phi::DataType::INT64),
                        phi::DataTypeToString(logits_lengths_dtype)));
  PADDLE_ENFORCE_EQ(label_lengths_dtype == phi::DataType::INT32 ||
                        label_lengths_dtype == phi::DataType::INT64,
                    true,
                    phi::errors::InvalidArgument(
                        "The data type of Input(label_lengths) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        phi::DataTypeToString(phi::DataType::INT32),
                        phi::DataTypeToString(phi::DataType::INT64),
                        phi::DataTypeToString(label_lengths_dtype)));

  // Step 1: LogSoftmax
  phi::DenseTensor log_probs;
  log_probs.set_meta(logits.meta());
  log_probs.Resize(logits.dims());
  custom_kernel::LogSoftmaxKernel<T, Context>(dev_ctx, logits, 2, &log_probs);

  // Step 2: CtcLoss Forward
  std::vector<int64_t> logits_lengths_vec;
  std::vector<int64_t> label_lengths_vec;
  if (logits_lengths_dtype == phi::DataType::INT32) {
    auto logits_lengths_vec_ =
        get_new_data_from_tensor<int>(dev_ctx, logits_lengths.get_ptr());
    auto label_lengths_vec_ =
        get_new_data_from_tensor<int>(dev_ctx, label_lengths.get_ptr());
    logits_lengths_vec = std::vector<int64_t>(logits_lengths_vec_.begin(),
                                              logits_lengths_vec_.end());
    label_lengths_vec = std::vector<int64_t>(label_lengths_vec_.begin(),
                                             label_lengths_vec_.end());
  } else {
    logits_lengths_vec =
        get_new_data_from_tensor<int64_t>(dev_ctx, logits_lengths.get_ptr());
    label_lengths_vec =
        get_new_data_from_tensor<int64_t>(dev_ctx, label_lengths.get_ptr());
  }

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

  phi::DenseTensor warpctc_label;
  custom_kernel::UnpadLabel<Context>(
      dev_ctx, int64_label, blank, label_lengths_vec, &warpctc_label);

  int64_t max_label_length =
      *std::max_element(label_lengths_vec.begin(), label_lengths_vec.end());
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
  phi::DenseTensor raw_loss;
  raw_loss.set_meta(loss->meta());
  dev_ctx.template Alloc<T>(&raw_loss);

  bool zero_infinity = true;
  EXEC_NPU_CMD(aclnnCtcLoss,
               dev_ctx,
               log_probs,
               warpctc_label,
               logits_lengths_vec,
               label_lengths_vec,
               blank,
               zero_infinity,
               raw_loss,
               log_alpha);

  // Step 3: Generate zero_inf_mask
  phi::DenseTensor zero_inf_mask;
  phi::DenseTensorMeta zero_inf_mask_meta = {phi::DataType::BOOL,
                                             label_lengths.get_ptr()->dims()};
  zero_inf_mask.set_meta(zero_inf_mask_meta);
  if (label_lengths_dtype == phi::DataType::INT32) {
    custom_kernel::GenZeroInfMask<int, Context>(dev_ctx,
                                                int64_label,
                                                *label_lengths.get_ptr(),
                                                *logits_lengths.get_ptr(),
                                                &zero_inf_mask);
  } else {
    custom_kernel::GenZeroInfMask<int64_t, Context>(dev_ctx,
                                                    int64_label,
                                                    *label_lengths.get_ptr(),
                                                    *logits_lengths.get_ptr(),
                                                    &zero_inf_mask);
  }

  // Step 4: Apply zero_inf_mask to raw_loss
  phi::DenseTensor zeros;
  zeros.set_meta(raw_loss.meta());
  custom_kernel::FullLike<T, Context>(
      dev_ctx, *loss, phi::Scalar(0.0), loss->dtype(), &zeros);
  custom_kernel::WhereKernel<T, Context>(
      dev_ctx, zero_inf_mask, raw_loss, zeros, loss);

  if (warpctcgrad != nullptr) {
    // Step 3: CtcLoss Backward (Compute loss and gradient in one call, as
    // CPU/GPU implementation does)
    phi::DenseTensor ones, log_probs_grad;
    ones.set_meta(loss->meta());
    log_probs_grad.set_meta(log_probs.meta());
    custom_kernel::FullLike<T, Context>(
        dev_ctx, *loss, phi::Scalar(1.0), loss->dtype(), &ones);
    log_probs_grad.Resize(log_probs.dims());
    custom_kernel::FullLike<T, Context>(
        dev_ctx, log_probs, phi::Scalar(0.0), loss->dtype(), &log_probs_grad);

    EXEC_NPU_CMD(aclnnCtcLossBackward,
                 dev_ctx,
                 ones,
                 log_probs,
                 warpctc_label,
                 logits_lengths_vec,
                 label_lengths_vec,
                 *loss,
                 log_alpha,
                 blank,
                 zero_infinity,
                 log_probs_grad);

    // Step 5: LogSoftmax Backward
    phi::DenseTensor logits_grad;
    logits_grad.set_meta(logits.meta());
    custom_kernel::LogSoftmaxGradKernel<T, Context>(
        dev_ctx, log_probs, log_probs_grad, 2, &logits_grad);

    // Step 6: Apply zero_inf_mask to logits_grad
    phi::DenseTensor zero_inf_mask_transform;
    phi::DenseTensorMeta zero_inf_mask_transform_meta = {logits_grad.dtype(),
                                                         zero_inf_mask.dims()};
    zero_inf_mask_transform.set_meta(zero_inf_mask_transform_meta);
    custom_kernel::CastKernel<bool, Context>(
        dev_ctx, zero_inf_mask, logits_grad.dtype(), &zero_inf_mask_transform);
    warpctcgrad->Resize(
        phi::make_ddim({max_sequence_length, num_sequences, sequence_width}));
    custom_kernel::MultiplyRawKernel<T, Context>(
        dev_ctx, logits_grad, zero_inf_mask_transform, 1, warpctcgrad);
  }
}

template <typename T, typename Context>
void WarpctcGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& logits,
                       const paddle::optional<phi::DenseTensor>& logits_lengths,
                       const phi::DenseTensor& warpctcgrad,
                       const phi::DenseTensor& loss_grad,
                       int blank,
                       bool norm_by_times,
                       phi::DenseTensor* logits_grad) {
  bool has_logits_lengths = logits_lengths.is_initialized();
  if (!has_logits_lengths) {
    PADDLE_THROW(phi::errors::External(
        "NPU only support logits_lengths is_initialized"));
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
