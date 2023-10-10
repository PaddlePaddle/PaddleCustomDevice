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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

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
  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(
        phi::errors::External("MLU only support logits_length is_initialized"));
  }
  bool has_labels_length = labels_length.is_initialized();
  if (!has_labels_length) {
    PADDLE_THROW(
        phi::errors::External("MLU only support labels_length is_initialized"));
  }

  int max_sequence_length = logits.dims()[0];
  int num_sequences = logits.dims()[1];
  int sequence_width = logits.dims()[2];
  int max_target_seq_length = label.dims()[1];
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
  PADDLE_ENFORCE_EQ(logits_length_dtype == DataType::INT32 ||
                        logits_length_dtype == DataType::INT64,
                    true,
                    phi::errors::InvalidArgument(
                        "The data type of Input(logits_length) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64),
                        DataTypeToString(logits_length_dtype)));
  PADDLE_ENFORCE_EQ(labels_length_dtype == DataType::INT32 ||
                        labels_length_dtype == DataType::INT64,
                    true,
                    phi::errors::InvalidArgument(
                        "The data type of Input(labels_length) should be "
                        "either %s or %s, "
                        "but received %s. ",
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64),
                        DataTypeToString(labels_length_dtype)));

  warpctcgrad->Resize(
      phi::make_ddim({max_sequence_length, num_sequences, sequence_width}));
  dev_ctx.template Alloc<T>(warpctcgrad);
  T* warpctcgrad_data = warpctcgrad->data<T>();

  int sm_workspace, lm_workspace;
  int64_t max_S = 2 * max_target_seq_length + 1;
  if (warpctcgrad_data == nullptr) {
    sm_workspace = sizeof(T) * sequence_width +
                   sizeof(int) * max_target_seq_length + sizeof(int);
    lm_workspace = 2 * max_S * sizeof(T) + 2 * max_S * sizeof(int);
  } else {
    sm_workspace = sizeof(T) * sequence_width +
                   sizeof(int) * max_target_seq_length + sizeof(int);
    lm_workspace = 4 * max_S * sizeof(T) + 2 * max_S * sizeof(int) +
                   sequence_width * sizeof(T);
  }

  PADDLE_ENFORCE_LE(
      sm_workspace + lm_workspace,
      256 * 1024,
      phi::errors::InvalidArgument(
          "Input size should be equal or less than %d for MLU warpctc kernel, "
          "but size %d is received. ",
          256 * 1024,
          sm_workspace + lm_workspace));

  loss->Resize(phi::make_ddim({num_sequences}));
  dev_ctx.template Alloc<T>(loss);

  auto logits_length_tensor = logits_length.get();
  auto labels_length_tensor = labels_length.get();
  MLUCnnlTensorDesc logits_length_desc(logits_length_tensor);
  MLUCnnlTensorDesc labels_length_desc(labels_length_tensor);

  Tensor logits_length_int32_tensor, labels_length_int32_tensor;
  logits_length_int32_tensor.Resize(logits_length.get_ptr()->dims());
  labels_length_int32_tensor.Resize(labels_length.get_ptr()->dims());
  dev_ctx.template Alloc<int>(&logits_length_int32_tensor);
  dev_ctx.template Alloc<int>(&labels_length_int32_tensor);
  MLUCnnlTensorDesc logits_length_int32_desc(logits_length_int32_tensor);
  MLUCnnlTensorDesc labels_length_int32_desc(labels_length_int32_tensor);

  int temp_max_label_length = 0;
  if (logits_length_dtype == DataType::INT64 &&
      labels_length_dtype == DataType::INT64) {
    // cast logits_length from int64 to int32
    auto cast_int64_to_int32_type =
        GetCastDataType(logits_length_dtype, DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_int64_to_int32_type,
                  logits_length_desc.get(),
                  GetBasePtr(&logits_length_tensor),
                  logits_length_int32_desc.get(),
                  GetBasePtr(&logits_length_int32_tensor));

    // cast labels_length from int64 to int32
    MLUCnnl::Cast(dev_ctx,
                  cast_int64_to_int32_type,
                  labels_length_desc.get(),
                  GetBasePtr(&labels_length_tensor),
                  labels_length_int32_desc.get(),
                  GetBasePtr(&labels_length_int32_tensor));
    logits_length_tensor = logits_length_int32_tensor;
    labels_length_tensor = labels_length_int32_tensor;

    std::vector<int64_t> vec_labels_length;
    TensorToVector(dev_ctx, *labels_length, dev_ctx, &vec_labels_length);
    dev_ctx.Wait();
    for (int i = 0; i < vec_labels_length.size(); ++i) {
      if (vec_labels_length[i] > temp_max_label_length)
        temp_max_label_length = vec_labels_length[i];
    }
  } else {
    std::vector<int> vec_labels_length;
    TensorToVector(dev_ctx, *labels_length, dev_ctx, &vec_labels_length);
    dev_ctx.Wait();
    for (int i = 0; i < vec_labels_length.size(); ++i) {
      if (vec_labels_length[i] > temp_max_label_length)
        temp_max_label_length = vec_labels_length[i];
    }
  }
  int max_label_length = static_cast<int>(temp_max_label_length);
  cnnlCTCLossNormalizationMode_t norm_mode = CNNL_NONE_NORMALIZATION;
  cnnlCTCLossReduceMode_t reduce_mode = CNNL_REDUCE_MODE_NONE;
  cnnlCTCLossZeroInfinityMode_t infinity_mode = CNNL_ZERO_INFINITY;
  MLUCnnlTensorDesc logits_desc(logits, CNNL_LAYOUT_TNC, CNNL_DTYPE_FLOAT);
  MLUCnnlTensorDesc label_desc(label);
  MLUCnnlTensorDesc loss_desc(*loss);
  MLUCnnlTensorDesc grads_desc(*warpctcgrad, CNNL_LAYOUT_TNC, CNNL_DTYPE_FLOAT);
  MLUCnnl::CTCLoss(dev_ctx,
                   norm_mode,
                   reduce_mode,
                   infinity_mode,
                   blank,
                   max_sequence_length,
                   max_label_length,
                   logits_desc.get(),
                   GetBasePtr(&logits),
                   label_desc.get(),
                   GetBasePtr(&label),
                   logits_length_int32_desc.get(),
                   GetBasePtr(&logits_length_tensor),
                   labels_length_int32_desc.get(),
                   GetBasePtr(&labels_length_tensor),
                   loss_desc.get(),
                   GetBasePtr(loss),
                   grads_desc.get(),
                   GetBasePtr(warpctcgrad));
  loss->Resize(phi::make_ddim({num_sequences, 1}));
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
  dev_ctx.template Alloc<T>(logits_grad);

  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(
        phi::errors::External("MLU only support logits_length is_initialized"));
  }
  int max_seq_length = warpctcgrad.dims()[0];  // Tmax
  int num_sequences = warpctcgrad.dims()[1];   // B
  int seq_width = warpctcgrad.dims()[2];       // D
  // B
  auto logits_len_e = *logits_length;
  // (B, 1)
  auto loss_grad_e = loss_grad;
  // (T, B, D)
  auto warpctcgrad_e = warpctcgrad;
  auto logits_grad_e = *logits_grad;
  logits_grad_e.Resize({1, num_sequences, 1});

  Tensor loss_grad_e_tmp;
  loss_grad_e_tmp.Resize(
      phi::make_ddim({max_seq_length, num_sequences, seq_width}));
  dev_ctx.template Alloc<T>(&logits_grad_e);
  dev_ctx.template Alloc<T>(&loss_grad_e_tmp);
  MLUCnnlTensorDesc logits_grad_e_tmp_desc(loss_grad_e_tmp);
  MLUCnnlTensorDesc logits_grad_e_desc(logits_grad_e);

  MLUCnnl::BroadcastTo(dev_ctx,
                       logits_grad_e_desc.get(),
                       GetBasePtr(&logits_grad_e),
                       logits_grad_e_tmp_desc.get(),
                       GetBasePtr(&loss_grad_e_tmp));
  Tensor logits_g;
  logits_g.Resize(logits_grad->dims());
  dev_ctx.template Alloc<T>(&logits_g);
  MLUCnnlTensorDesc logits_g_desc(logits_g);
  MLUCnnlTensorDesc warpctcgrad_e_desc(warpctcgrad_e);

  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    warpctcgrad_e_desc.get(),
                    GetBasePtr(&warpctcgrad_e),
                    logits_grad_e_tmp_desc.get(),
                    GetBasePtr(&loss_grad_e_tmp),
                    logits_g_desc.get(),
                    GetBasePtr(&logits_g),
                    ToCnnlDataType<T>());

  if (!norm_by_times) {
    *logits_grad = logits_g;
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    warpctc, mlu, ALL_LAYOUT, custom_kernel::WarpctcKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    warpctc_grad, mlu, ALL_LAYOUT, custom_kernel::WarpctcGradKernel, float) {}
