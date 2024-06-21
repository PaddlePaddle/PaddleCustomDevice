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
#include <numeric>

#include "funcs/sdaa_baseop.h"
#include "funcs/slice_utils.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Type>
bool IsContinuous(const Type& weight_list) {
  bool continuous = true;
  for (size_t i = 0; i < weight_list.size() - 1; ++i) {
    auto* in_data = weight_list[i]->template data<T>();
    auto* in_after_data = weight_list[i + 1]->template data<T>();
    auto in_size = weight_list[i]->numel();
    bool temp = in_data + in_size == in_after_data;
    continuous = continuous && temp;
  }
  return continuous;
}

template <typename T>
void WeightToTensor(const Context& dev_ctx,
                    const std::vector<const phi::DenseTensor*>& weight_list,
                    phi::DenseTensor* weight) {
  auto weight_data = weight->data<T>();
  int weight_offset = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    const T* in_data = weight_list[i]->data<T>();
    auto in_size = weight_list[i]->numel();

    AsyncMemCpyD2D(nullptr,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   weight_data + weight_offset,
                   in_data,
                   in_size * sizeof(T));

    weight_offset += in_size;
  }
}

template <typename T>
size_t GetWeightNum(const std::vector<const phi::DenseTensor*>& weight_list) {
  size_t weight_num = std::accumulate(
      weight_list.begin(),
      weight_list.end(),
      0,
      [](int64_t num, const phi::DenseTensor* t) { return num + t->numel(); });

  return weight_num;
}

template <typename T, typename Context>
tecodnnRNNDescriptor_t GetTecodnnRnnDesc(const Context& dev_ctx,
                                         tecodnnHandle_t tecodnnHandle,
                                         int hidden_size,
                                         int num_layers,
                                         int seed,
                                         float drop_prob,
                                         bool is_bidirec,
                                         const std::string& rnn_mode,
                                         const std::string& input_mode,
                                         const std::string& rnn_algo) {
  tecodnnRNNInputMode_t InputMode = TECODNN_LINEAR_INPUT;
  if (input_mode == "skip") {
    InputMode = TECODNN_SKIP_INPUT;
  }

  tecodnnDirectionMode_t DirectionMode = TECODNN_UNIDIRECTIONAL;
  if (is_bidirec) {
    DirectionMode = TECODNN_BIDIRECTIONAL;
  }

  tecodnnRNNMode_t RnnMode = TECODNN_LSTM;
  if (rnn_mode == "LSTM")
    RnnMode = TECODNN_LSTM;
  else if (rnn_mode == "GRU")
    RnnMode = TECODNN_GRU;
  else if (rnn_mode == "RNN_RELU")
    RnnMode = TECODNN_RNN_RELU;
  else if (rnn_mode == "RNN_TANH")
    RnnMode = TECODNN_RNN_TANH;
  else
    PADDLE_THROW(phi::errors::InvalidArgument(
        "rnn_mode should be LSTM, GRU, RNN_RELU or RNN_TANH, but received: "
        "%s.",
        rnn_mode));

  tecodnnRNNAlgo_t RnnAlgo =
      TECODNN_RNN_ALGO_STANDARD;  // TECODNN only support this algorithm

  tecodnnDataType_t dt =
      sdaa_ops::ToTecodnnDataType(phi::CppTypeToDataType<T>::Type());

  // set dropout descriptor
  tecodnnDropoutDescriptor_t DropoutDesc;
  TECODNN_CHECK(tecodnnCreateDropoutDescriptor(&DropoutDesc));
  // set dropout states
  size_t act_statesSize = 4 * 1024 * sizeof(int);
  // when statesSize is fixed, will use DropoutGetStatesSize() func.
  TECODNN_CHECK(tecodnnDropoutGetStatesSize(tecodnnHandle, &act_statesSize));
  phi::DenseTensorMeta meta = {phi::DataType::INT8,
                               {static_cast<int>(act_statesSize)}};
  phi::DenseTensor states;
  states.set_meta(meta);
  dev_ctx.template Alloc<int8_t>(&states);
  TECODNN_CHECK(tecodnnSetDropoutDescriptor(DropoutDesc,
                                            tecodnnHandle,
                                            drop_prob,
                                            states.data(),
                                            act_statesSize,
                                            seed));

  // set rnn descriptor
  tecodnnRNNDescriptor_t RnnDesc;
  TECODNN_CHECK(tecodnnCreateRNNDescriptor(&RnnDesc));
  TECODNN_CHECK(tecodnnSetRNNDescriptor(tecodnnHandle,
                                        RnnDesc,
                                        hidden_size,
                                        num_layers,
                                        DropoutDesc,
                                        InputMode,
                                        DirectionMode,
                                        RnnMode,
                                        RnnAlgo,
                                        dt));
  TECODNN_CHECK(tecodnnDestroyDropoutDescriptor(DropoutDesc));

  return RnnDesc;
}

template <typename T, typename Context>
void doRnnForward(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& hx,
                  const phi::DenseTensor& cx,
                  const std::vector<const phi::DenseTensor*>& weight_list,
                  int batch_size,
                  int input_size,
                  int direction_num,
                  int hidden_size,
                  int num_layers,
                  int seed,
                  int sequence_len,
                  float dropout_prob,
                  bool is_bidirec,
                  bool is_test,
                  const std::string& mode,
                  phi::DenseTensor* y,
                  phi::DenseTensor* hy,
                  phi::DenseTensor* cy,
                  phi::DenseTensor* dropout_state,
                  phi::DenseTensor* reserve) {
  std::string input_mode = "linear";
  std::string rnn_algo = "standard";

  std::vector<int> hx_dims = {
      direction_num * num_layers, batch_size, hidden_size};

  size_t workspace_size, reserve_size, weightspace_size;

  int weight_num = static_cast<int>(GetWeightNum<T>(weight_list));

  phi::DenseTensor weight_whole;
  void* w_data = nullptr;
  bool continuous =
      IsContinuous<T, std::vector<const phi::DenseTensor*>>(weight_list);
  if (!continuous) {
    VLOG(2) << "If the memory space of the Input WeightList is not continuous, "
               "less efficient calculation will be called. Please call "
               "flatten_parameters() to make the input memory continuous.";

    weight_whole.Resize(phi::make_ddim({weight_num}));
    dev_ctx.template Alloc<T>(&weight_whole);

    WeightToTensor<T>(dev_ctx, weight_list, &weight_whole);

    w_data = weight_whole.data();
  } else {
    w_data = const_cast<void*>(weight_list[0]->data());
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnRNNDescriptor_t RnnDesc = GetTecodnnRnnDesc<T, Context>(dev_ctx,
                                                                 tecodnnHandle,
                                                                 hidden_size,
                                                                 num_layers,
                                                                 seed,
                                                                 dropout_prob,
                                                                 is_bidirec,
                                                                 mode,
                                                                 input_mode,
                                                                 rnn_algo);

  tecodnnDataType_t tecodnn_dt = sdaa_ops::ToTecodnnDataType(x.dtype());
  tecodnnRNNDataLayout_t RnnLayout = TECODNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  tecodnnForwardMode_t RnnFwdMode =
      is_test ? TECODNN_FWD_MODE_INFERENCE : TECODNN_FWD_MODE_TRAINING;
  tecodnnRNNDataDescriptor_t x_Desc, y_Desc;
  TECODNN_CHECK(tecodnnCreateRNNDataDescriptor(&x_Desc));
  TECODNN_CHECK(tecodnnCreateRNNDataDescriptor(&y_Desc));
  TECODNN_CHECK(tecodnnSetRNNDataDescriptor(
      x_Desc,
      tecodnn_dt,
      RnnLayout,
      sequence_len,
      batch_size,
      input_size,
      nullptr,  // If variable batch size is supported, do not pass null.
      nullptr));
  TECODNN_CHECK(tecodnnSetRNNDataDescriptor(
      y_Desc,
      tecodnn_dt,
      RnnLayout,
      sequence_len,
      batch_size,
      direction_num * hidden_size,
      nullptr,  // If variable batch size is supported, do not pass null.
      nullptr));

  tecodnnTensorDescriptor_t hx_Desc = sdaa_ops::GetTecodnnTensorDesc(
      hx_dims, hx.dtype(), TensorFormat::Undefined);

  // get the workspace size and reserve size for rnn
  TECODNN_CHECK(tecodnnGetRNNTempSpaceSizes(tecodnnHandle,
                                            RnnDesc,
                                            RnnFwdMode,
                                            x_Desc,
                                            &workspace_size,
                                            &reserve_size));
  // get the weightspace size for rnn
  TECODNN_CHECK(tecodnnGetRNNWeightSpaceSize(
      tecodnnHandle, RnnDesc, x_Desc, &weightspace_size));

  PADDLE_ENFORCE_EQ(
      weightspace_size,
      weight_num * sizeof(T),
      phi::errors::InvalidArgument(
          "The sdaa rnn and setting weight size should be same."));

  phi::DenseTensor workspace_data;
  workspace_data.Resize(phi::make_ddim({static_cast<int64_t>(workspace_size)}));
  dev_ctx.template Alloc<uint8_t>(&workspace_data);
  sdaa_ops::doFillTensor<uint8_t>(
      dev_ctx, static_cast<uint8_t>(0), phi::DataType::UINT8, &workspace_data);

  reserve->Resize(phi::make_ddim({static_cast<int64_t>(reserve_size)}));
  dev_ctx.template Alloc<uint8_t>(reserve);

  TECODNN_CHECK(tecodnnRNNForward(
      tecodnnHandle,
      RnnDesc,
      RnnFwdMode,
      nullptr,  // If variable batch size is supported, do not pass null.
      x_Desc,
      x.data(),
      y_Desc,
      y->data(),
      hx_Desc,
      hx.data(),
      hy->data(),
      hx_Desc,
      cx.data(),
      cy->data(),
      weightspace_size,
      w_data,
      workspace_size,
      workspace_data.data(),
      reserve_size,
      reserve->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(hx_Desc));
  TECODNN_CHECK(tecodnnDestroyRNNDataDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyRNNDataDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyRNNDescriptor(RnnDesc));
}

template <typename T, typename Context>
void doRnnBackward(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& hx,
                   const phi::DenseTensor& cx,
                   const phi::DenseTensor& dhy,
                   const phi::DenseTensor& dcy,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& out_grad,
                   const phi::DenseTensor& reserve,
                   const std::vector<const phi::DenseTensor*>& weight_list,
                   int batch_size,
                   int input_size,
                   int direction_num,
                   int hidden_size,
                   int num_layers,
                   int seed,
                   int sequence_len,
                   float dropout_prob,
                   bool is_bidirec,
                   bool is_test,
                   const std::string& mode,
                   phi::DenseTensor* x_grad,
                   phi::DenseTensor* dhx,
                   phi::DenseTensor* dcx,
                   std::vector<phi::DenseTensor*> weight_grad_list) {
  std::string input_mode = "linear";
  std::string rnn_algo = "standard";

  std::vector<int> hx_dims = {
      direction_num * num_layers, batch_size, hidden_size};

  size_t workspace_size, reserve_size, weightspace_size;

  int64_t weight_num = static_cast<int64_t>(GetWeightNum<T>(weight_list));
  bool continuous =
      IsContinuous<T, std::vector<const phi::DenseTensor*>>(weight_list);

  phi::DenseTensor weight_whole;
  void* w_data = nullptr;
  if (!continuous) {
    VLOG(2) << "If the memory space of the Input WeightList is not continuous, "
               "less efficient calculation will be called. Please call "
               "flatten_parameters() to make the input memory continuous.";

    weight_whole.Resize(phi::make_ddim({weight_num}));
    dev_ctx.template Alloc<T>(&weight_whole);

    WeightToTensor<T>(dev_ctx, weight_list, &weight_whole);

    w_data = weight_whole.data();
  } else {
    w_data = const_cast<void*>(weight_list[0]->data());
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnRNNDescriptor_t RnnDesc = GetTecodnnRnnDesc<T, Context>(dev_ctx,
                                                                 tecodnnHandle,
                                                                 hidden_size,
                                                                 num_layers,
                                                                 seed,
                                                                 dropout_prob,
                                                                 is_bidirec,
                                                                 mode,
                                                                 input_mode,
                                                                 rnn_algo);

  tecodnnDataType_t tecodnn_dt = sdaa_ops::ToTecodnnDataType(x.dtype());
  tecodnnRNNDataLayout_t RnnLayout = TECODNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  tecodnnForwardMode_t RnnFwdMode =
      is_test ? TECODNN_FWD_MODE_INFERENCE : TECODNN_FWD_MODE_TRAINING;
  tecodnnRNNDataDescriptor_t x_Desc, y_Desc;
  TECODNN_CHECK(tecodnnCreateRNNDataDescriptor(&x_Desc));
  TECODNN_CHECK(tecodnnCreateRNNDataDescriptor(&y_Desc));
  TECODNN_CHECK(tecodnnSetRNNDataDescriptor(
      x_Desc,
      tecodnn_dt,
      RnnLayout,
      sequence_len,
      batch_size,
      input_size,
      nullptr,  // If variable batch size is supported, do not pass null.
      nullptr));
  TECODNN_CHECK(tecodnnSetRNNDataDescriptor(
      y_Desc,
      tecodnn_dt,
      RnnLayout,
      sequence_len,
      batch_size,
      direction_num * hidden_size,
      nullptr,  // If variable batch size is supported, do not pass null.
      nullptr));

  tecodnnTensorDescriptor_t hx_Desc = sdaa_ops::GetTecodnnTensorDesc(
      hx_dims, hx.dtype(), TensorFormat::Undefined);

  // get the workspace size and reserve size for rnn
  TECODNN_CHECK(tecodnnGetRNNTempSpaceSizes(tecodnnHandle,
                                            RnnDesc,
                                            RnnFwdMode,
                                            x_Desc,
                                            &workspace_size,
                                            &reserve_size));
  // get the weightspace size for rnn
  TECODNN_CHECK(tecodnnGetRNNWeightSpaceSize(
      tecodnnHandle, RnnDesc, x_Desc, &weightspace_size));

  phi::DenseTensor workspace_data;
  workspace_data.Resize(phi::make_ddim({static_cast<int64_t>(workspace_size)}));
  dev_ctx.template Alloc<uint8_t>(&workspace_data);
  sdaa_ops::doFillTensor<uint8_t>(
      dev_ctx, static_cast<uint8_t>(0), phi::DataType::UINT8, &workspace_data);

  uint8_t* reserve_data = const_cast<uint8_t*>(reserve.data<uint8_t>());

  if (x_grad) {
    void* dhx_data_ptr = nullptr;
    void* dcx_data_ptr = nullptr;
    if (dhx) {
      dhx_data_ptr = dhx->data();
    }
    if (dcx) {
      dcx_data_ptr = dcx->data();
    }

    TECODNN_CHECK(tecodnnRNNBackwardData(
        tecodnnHandle,
        RnnDesc,
        nullptr,  // If variable batch size is supported, do not pass null.
        y_Desc,
        out.data(),
        out_grad.data(),
        x_Desc,
        x_grad->data(),
        hx_Desc,
        hx.data(),
        dhy.data(),
        dhx_data_ptr,
        hx_Desc,
        cx.data(),
        dcy.data(),
        dcx_data_ptr,
        weightspace_size,
        w_data,
        workspace_size,
        workspace_data.data(),
        reserve_size,
        reserve_data));
  }
  if (!weight_grad_list.empty()) {
    // 1. Allocate a contiguous block of memory space to the
    // tecodnnRNNBackwardWeights.
    phi::DenseTensor weight_grad;
    weight_grad.Resize(phi::make_ddim({weight_num}));
    dev_ctx.template Alloc<T>(&weight_grad);
    sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), &weight_grad);

    tecodnnWgradMode_t tecodnn_wgrad_mode = TECODNN_WGRAD_MODE_ADD;

    TECODNN_CHECK(tecodnnRNNBackwardWeights(
        tecodnnHandle,
        RnnDesc,
        tecodnn_wgrad_mode,
        nullptr,  // If variable batch size is supported, do not pass null.
        x_Desc,
        x.data(),
        hx_Desc,
        hx.data(),
        y_Desc,
        out.data(),
        weightspace_size,
        weight_grad.data(),
        workspace_size,
        workspace_data.data(),
        reserve_size,
        reserve_data));

    // 2. Copy the values in the continuous space to the origin separate space.
    int64_t offset = 0;
    for (size_t i = 0; i < weight_grad_list.size(); ++i) {
      int64_t len = weight_grad_list[i]->numel();
      auto dim = weight_grad_list[i]->dims();
      auto sub_weight_grad =
          custom_kernel::Slice(weight_grad, offset, offset + len);
      phi::Copy(dev_ctx,
                sub_weight_grad,
                dev_ctx.GetPlace(),
                false,
                weight_grad_list[i]);
      weight_grad_list[i]->Resize(dim);
      offset += len;
    }
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(hx_Desc));
  TECODNN_CHECK(tecodnnDestroyRNNDataDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyRNNDataDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyRNNDescriptor(RnnDesc));
}

template <typename T, typename Context>
void RnnKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<const phi::DenseTensor*>& pre_state,
               const std::vector<const phi::DenseTensor*>& weight_list,
               const paddle::optional<phi::DenseTensor>& sequence_length,
               float dropout_prob,
               bool is_bidirec,
               int input_size,
               int hidden_size,
               int num_layers,
               const std::string& mode,
               int seed,
               bool is_test,
               phi::DenseTensor* out,
               phi::DenseTensor* dropout_state,
               std::vector<phi::DenseTensor*> state,
               phi::DenseTensor* reserve) {
  VLOG(4) << "CALL SDAA RnnKernel";

  PADDLE_ENFORCE_EQ(
      mode,
      "LSTM",
      phi::errors::InvalidArgument("sdaa currently only support LSTM, "
                                   "but got %s.",
                                   mode));

  PADDLE_ENFORCE_EQ(
      dropout_prob,
      0.f,
      phi::errors::InvalidArgument(
          "sdaa currently only support dropout_prob = 0, but got %s.",
          dropout_prob));

  PADDLE_ENFORCE_EQ(
      is_test,
      false,
      phi::errors::InvalidArgument("sdaa currently only support training for "
                                   "RNN Op, but the is_test value is %s.",
                                   is_test));

  // Input and output
  auto init_h = pre_state[0];  // -> hx
  auto init_c = pre_state[1];  // -> cx, only in LSTM
  auto last_h = state[0];      // -> hy
  auto last_c = state[1];      // -> cy, only in LSTM

  // init the output and allocate the memory
  dev_ctx.template Alloc<T>(out);     // y in sdaa
  dev_ctx.template Alloc<T>(last_h);  // hy in sdaa
  dev_ctx.template Alloc<T>(last_c);  // cy in sdaa

  // allocate memory for dropout_state
  dropout_state->Resize(out->dims());
  dev_ctx.template Alloc<uint8_t>(dropout_state);
  sdaa_ops::doFillTensor<uint8_t>(
      dev_ctx, static_cast<uint8_t>(1), dropout_state->dtype(), dropout_state);

  // check shape
  const int in_out_dim_num = x.dims().size();
  const int& seq_len = x.dims()[0];
  const int& batch_size = x.dims()[1];
  const int& input_size_local = x.dims()[2];
  const int& direction_num = is_bidirec ? 2 : 1;

  std::vector<int> seq_len_vec(batch_size, seq_len);
  if (sequence_length.is_initialized()) {
    PADDLE_THROW(
        phi::errors::InvalidArgument("TECODNN not support the variable "
                                     "sequence lengths in a batch."));
  }

  PADDLE_ENFORCE_EQ(
      init_h->dims()[0],
      num_layers * direction_num,
      phi::errors::InvalidArgument("The num_layers of in RNN layer must"
                                   "be the same as first dim of init hidden, "
                                   "but received num_layers:%d, dim: %d",
                                   num_layers,
                                   init_h->dims()[0]));

  PADDLE_ENFORCE_EQ(
      init_c->dims()[0],
      num_layers * direction_num,
      phi::errors::InvalidArgument("The num_layers of in RNN layer must"
                                   "be the same as first dim of cell state"
                                   "hidden, but received num_layers:%d, dim:%d",
                                   num_layers,
                                   init_c->dims()[0]));

  doRnnForward<T, Context>(dev_ctx,
                           x,
                           *init_h,
                           *init_c,
                           weight_list,
                           batch_size,
                           input_size_local,
                           direction_num,
                           hidden_size,
                           num_layers,
                           seed,
                           seq_len,
                           dropout_prob,
                           is_bidirec,
                           is_test,
                           mode,
                           out,
                           last_h,
                           last_c,
                           dropout_state,
                           reserve);
}

template <typename T, typename Context>
void RnnGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const std::vector<const phi::DenseTensor*>& pre_state,
                   const std::vector<const phi::DenseTensor*>& weight_list,
                   const paddle::optional<phi::DenseTensor>& sequence_length,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dropout_state,
                   const phi::DenseTensor& reserve,
                   const phi::DenseTensor& out_grad,
                   const std::vector<const phi::DenseTensor*>& state_grad,
                   float dropout_prob,
                   bool is_bidirec,
                   int input_size,
                   int hidden_size,
                   int num_layers,
                   const std::string& mode,
                   int seed,
                   bool is_test,
                   phi::DenseTensor* x_grad,
                   std::vector<phi::DenseTensor*> pre_state_grad,
                   std::vector<phi::DenseTensor*> weight_grad_list) {
  VLOG(4) << "CALL SDAA RnnGradKernel";

  PADDLE_ENFORCE_EQ(
      mode,
      "LSTM",
      phi::errors::InvalidArgument("sdaa currently only support LSTM, "
                                   "but got %s.",
                                   mode));

  auto init_h = pre_state[0];        // -> hx
  auto init_c = pre_state[1];        // -> cx
  auto last_h_grad = state_grad[0];  // -> dhy
  auto last_c_grad = state_grad[1];  // -> dcy

  phi::DenseTensor* init_h_grad = nullptr;
  phi::DenseTensor* init_c_grad = nullptr;
  if (pre_state_grad.size() > 0) {    // has gradient
    init_h_grad = pre_state_grad[0];  // -> dhx
    init_c_grad = pre_state_grad[1];  // -> dcx
  }

  // check shape
  const int in_out_dim_numin_out_dim_num = x.dims().size();
  const int& seq_len = x.dims()[0];
  const int& batch_size = x.dims()[1];
  const int& input_size_local = x.dims()[2];
  const int& direction_num = is_bidirec ? 2 : 1;

  std::vector<int> seq_len_vec(batch_size, seq_len);
  if (sequence_length.is_initialized()) {
    PADDLE_THROW(
        phi::errors::InvalidArgument("TECODNN not support the variable "
                                     "sequence lengths in a batch."));
  }

  PADDLE_ENFORCE_EQ(
      init_h->dims()[0],
      num_layers * direction_num,
      phi::errors::InvalidArgument("The num_layers of in RNN layer must"
                                   "be the same as first dim of init hidden, "
                                   "but received num_layers:%d, dim: %d",
                                   num_layers,
                                   init_h->dims()[0]));

  PADDLE_ENFORCE_EQ(
      init_c->dims()[0],
      num_layers * direction_num,
      phi::errors::InvalidArgument("The num_layers of in RNN layer must"
                                   "be the same as first dim of cell state"
                                   "hidden, but received num_layers:%d, dim:%d",
                                   num_layers,
                                   init_c->dims()[0]));

  phi::DenseTensor input_grad_value;
  if (!x_grad) {
    x_grad = &input_grad_value;
    x_grad->Resize(x.dims());
  }
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
  }

  if (init_h_grad) {
    dev_ctx.template Alloc<T>(init_h_grad);
  }
  if (init_c_grad) {
    dev_ctx.template Alloc<T>(init_c_grad);
  }

  for (size_t i = 0; i < weight_grad_list.size(); i++) {
    dev_ctx.template Alloc<T>(weight_grad_list[i]);
  }

  doRnnBackward<T, Context>(dev_ctx,
                            x,
                            *init_h,
                            *init_c,
                            *last_h_grad,
                            *last_c_grad,
                            out,
                            out_grad,
                            reserve,
                            weight_list,
                            batch_size,
                            input_size_local,
                            direction_num,
                            hidden_size,
                            num_layers,
                            seed,
                            seq_len,
                            dropout_prob,
                            is_bidirec,
                            is_test,
                            mode,
                            x_grad,
                            init_h_grad,
                            init_c_grad,
                            weight_grad_list);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    rnn, sdaa, ALL_LAYOUT, custom_kernel::RnnKernel, float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}

PD_REGISTER_PLUGIN_KERNEL(
    rnn_grad, sdaa, ALL_LAYOUT, custom_kernel::RnnGradKernel, float) {}
