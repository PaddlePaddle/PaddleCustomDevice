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

template <typename TensorType, typename T>
void reset_parameter_vector(
    const std::vector<TensorType>& raw_params_vec,
    const int& num_layers,
    const bool& is_bidirec,
    std::vector<std::vector<std::pair<T*, size_t>>>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    params_vec->at(i).resize(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      using remove_cv_t = typename std::remove_cv<T>::type;
      params_vec->at(i)[j] = std::make_pair(
          const_cast<T*>(
              raw_params_vec[tensor_idx]->template data<remove_cv_t>()),
          raw_params_vec[tensor_idx]->numel() * sizeof(T));
    }
  }
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
               phi::DenseTensor* reserve){
    // Input
    auto init_h = pre_state[0];  // -> hx
    auto init_c = pre_state[1];  // -> cx
    auto last_h = state[0];
    auto last_c = state[1];

    // check shape
    const int in_out_dim_num = x.dims().size();
    const int& seq_len = x.dims()[0];  // time_step
    const int& batch_size = x.dims()[1];
    const int& input_dim = x.dims()[2];
    const int& direction_num = is_bidirec ? 2 : 1;
    int in_dim_arr[in_out_dim_num] = {seq_len, batch_size, input_dim};
    int out_dim_arr[in_out_dim_num] = {
        seq_len, batch_size, direction_num * hidden_size};
    int proj_size = hidden_size;

    std::vector<int> seq_len_vec(batch_size, seq_len);
    if (sequence_length.is_initialized()) {  // set seq_len if no padding, otherwise seq_len for
      // each element.
     custom_kernel::TensorToVector(dev_ctx, *sequence_length, dev_ctx, &seq_len_vec);
    }
    cnnlDirectionMode_t direction =
        is_bidirec ? CNNL_RNN_BIDIRECTIONAL : CNNL_RNN_UNIDIRECTIONAL;

    PADDLE_ENFORCE_EQ(
        mode,
        "LSTM",
        phi::errors::InvalidArgument(
            "MLU only support LSTM mode now, current mode is %s", mode));
    PADDLE_ENFORCE_EQ(
        num_layers,
        1,
        phi::errors::InvalidArgument(
            "MLU only support 1 num_layers, current num_layers is %s",
            num_layers));
    PADDLE_ENFORCE_EQ(
        init_h->dims()[0],
        num_layers * direction_num,
        phi::errors::InvalidArgument("The num_layers of in RNN layer must"
                                          " be the same as first dim of init "
                                          "hidden, but received num_layers:%d,"
                                          " dim:%d",
                                          num_layers,
                                          init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0],
        num_layers * direction_num,
        phi::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers,
            init_c->dims()[0]));

    // weightlist
    std::vector<std::vector<std::pair<T*, size_t>>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(weight_list, num_layers, is_bidirec, &parameter_lists);

    // init the output and allocate the memory  
    dev_ctx.template Alloc<T>(out);// -> y in cnnl
    dev_ctx.template Alloc<T>(last_h);// -> hy in cnnl
    dev_ctx.template Alloc<T>(last_c);// -> cy in cnnl

    MLUSeqDataDesc input_seq_data_desc(CNNL_SEQDATA_TNC,
                                       ToCnnlDataType(x.dtype()),
                                       in_out_dim_num,
                                       in_dim_arr,
                                       static_cast<int>(seq_len_vec.size()),
                                       seq_len_vec.data(),
                                       nullptr);
    MLUSeqDataDesc out_seq_data_desc(CNNL_SEQDATA_TNC,
                                     ToCnnlDataType(x.dtype()),
                                     in_out_dim_num,
                                     out_dim_arr,
                                     static_cast<int>(seq_len_vec.size()),
                                     seq_len_vec.data(),
                                     nullptr);
    MLUCnnlTensorDesc hx_desc(*init_h);
    MLUCnnlTensorDesc cx_desc(*init_c);

    MLURNNDesc rnn_desc(CNNL_LSTM,
                        CNNL_RNN_DOUBLE_BIAS,
                        direction,
                        CNNL_RNN_LINEAR_INPUT,
                        ToCnnlDataType(x.dtype()),
                        ToCnnlDataType(x.dtype()),
                        input_dim,
                        hidden_size,
                        /*projection*/ proj_size,
                        num_layers,
                        nullptr,
                        CNNL_RNN_PADDED_IO_DISABLED);
    rnn_desc.SetRNNMaskMode(CNNL_LSTM_MASK_ENABLED);

    // copy weight params
    size_t weightspace_size;
    Tensor weightspace;
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetRNNWeightSpaceSize(
        GetHandleFromCTX(dev_ctx), rnn_desc.get(), &weightspace_size));

    weightspace.Resize({static_cast<int64_t>(weightspace_size)});
    dev_ctx.template Alloc<T>(&weightspace);

    void* weightspace_ptr = dev_ctx.template Alloc<T>(&weightspace);
    auto w_x = parameter_lists[0][0];
    auto w_h = parameter_lists[0][1];
    auto b_x = parameter_lists[0][2];
    auto b_h = parameter_lists[0][3];
    auto actual_total_w_size =
        w_x.second + w_h.second + b_x.second + b_h.second;

    void* w_x_ptr = weightspace_ptr;
    void* w_h_ptr = static_cast<char*>(weightspace_ptr) + w_x.second;
    void* b_x_ptr =
        static_cast<char*>(weightspace_ptr) + w_x.second + w_h.second;
    void* b_h_ptr = static_cast<char*>(weightspace_ptr) + w_x.second +
                    w_h.second + b_x.second;

    MemCpyD2D(nullptr,
          w_x_ptr,
          w_x.first,
          w_x.second);

    MemCpyD2D(nullptr,
          w_h_ptr,
          w_h.first,
          w_h.second);
    MemCpyD2D(nullptr,
          b_x_ptr,
          b_x.first,
          b_x.second);
    MemCpyD2D(nullptr,
          b_h_ptr,
          b_h.first,
          b_h.second);

    if (is_bidirec) {
      auto bw_x = parameter_lists[0][4];
      auto bw_h = parameter_lists[0][5];
      auto bb_x = parameter_lists[0][6];
      auto bb_h = parameter_lists[0][7];
      void* bw_x_ptr =
          static_cast<char*>(weightspace_ptr) + actual_total_w_size;
      void* bw_h_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second;
      void* bb_x_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second + bw_h.second;
      void* bb_h_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second + bw_h.second +
                       bb_x.second;
      actual_total_w_size +=
          bw_x.second + bw_h.second + bb_x.second + bb_h.second;
    MemCpyD2D(nullptr,
          bw_x_ptr,
          bw_x.first,
          bw_x.second);
    MemCpyD2D(nullptr,
          bw_h_ptr,
          bw_h.first,
          bw_h.second);
    MemCpyD2D(nullptr,
          bb_x_ptr,
          bb_x.first,
          bb_x.second);
     MemCpyD2D(nullptr,
          bb_h_ptr,
          bb_h.first,
          bb_h.second);
    
    }

    PADDLE_ENFORCE_EQ(weightspace_size,
                      actual_total_w_size,
                      phi::errors::InvalidArgument(
                          "The weightsize doesn't match"
                          " weightspace_size:%d, actual_total_w_size:%d",
                          weightspace_size,
                          actual_total_w_size));

    // get reservespace_ptr
    int gate_num = 4;
    int hidden_data_idx = (num_layers - 1);
    hidden_data_idx += (gate_num + 1) * num_layers;
    const int& block_size = direction_num * seq_len * batch_size * hidden_size;
    reserve->Resize({hidden_data_idx, block_size});

    dev_ctx.template Alloc<T>(reserve);

    MLUCnnl::RNNForward(dev_ctx,
                        rnn_desc.get(),
                        seq_len_vec.data(),
                        weightspace_ptr,
                        weightspace_size,
                        input_seq_data_desc.get(),
                        GetBasePtr(&x),
                        out_seq_data_desc.get(),
                        GetBasePtr(out),
                        hx_desc.get(),
                        GetBasePtr(init_h),
                        GetBasePtr(last_h),
                        cx_desc.get(),
                        GetBasePtr(init_c),
                        GetBasePtr(last_c),
                        GetBasePtr(reserve));

    if (sequence_length) {
      // if has_seq_length, do mask out the output of cnnlRNNForwardTraining
      auto masked_mode = CNNL_MASKED_FILL;
      float off_value = 0.0f;

      Tensor on_value_tensor;
      Tensor masked_tensor;
      Tensor h_masked_tensor;
      on_value_tensor.Resize({1});
      masked_tensor.Resize({seq_len, batch_size, direction_num * hidden_size});
      h_masked_tensor.Resize({seq_len, batch_size, direction_num * hidden_size});

      dev_ctx.template Alloc<T>(&on_value_tensor);
      dev_ctx.template Alloc<int8_t>(&masked_tensor);
      int8_t* h_masked_ptr = dev_ctx.template HostAlloc<int8_t>(&h_masked_tensor);

      for (int t = 0; t < seq_len; ++t) {
        for (int n = 0; n < batch_size; ++n) {
          for (int c = 0; c < direction_num * hidden_size; ++c) {
            auto tmp_seq_len = seq_len_vec[n];
            auto offset = t * batch_size * direction_num * hidden_size +
                          n * direction_num * hidden_size + c;
            *(h_masked_ptr + offset) = t >= tmp_seq_len ? 1 : 0;
          }
        }
      }

      TensorCopy(dev_ctx, h_masked_tensor,false, &masked_tensor);
      dev_ctx.Wait();


      FillMLUTensorWithHostValue(dev_ctx, off_value, &on_value_tensor);
      MLUCnnlTensorDesc on_value_desc(on_value_tensor);
      MLUCnnlTensorDesc output_desc(*out);
      MLUCnnlTensorDesc masked_desc(masked_tensor);

      MLUCnnl::Mask(dev_ctx,
                    masked_mode,
                    output_desc.get(),
                    GetBasePtr(out),
                    masked_desc.get(),
                    GetBasePtr(&masked_tensor),
                    on_value_desc.get(),
                    GetBasePtr(&on_value_tensor),
                    output_desc.get(),
                    GetBasePtr(out),
                    nullptr);


    }
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
                   std::vector<phi::DenseTensor*> weight_grad_list){
    C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());

    PADDLE_ENFORCE_EQ(
        mode,
        "LSTM",
        phi::errors::InvalidArgument(
            "MLU only support LSTM mode now, current mode is %s", mode));

    auto init_h = pre_state[0];  // -> hx
    auto init_c = pre_state[1];  // -> cx
    auto last_h_grad = state_grad[0];  // -> dhy
    auto last_c_grad = state_grad[1];  // -> dcy

    Tensor* init_h_grad = nullptr;
    Tensor* init_c_grad = nullptr;
    if (pre_state_grad.size() > 0) {    // has gradient
      init_h_grad = pre_state_grad[0];  // -> dhx
      init_c_grad = pre_state_grad[1];  // -> dcx
    }

    // check shape
    const int in_out_dim_num = x.dims().size();
    const int& seq_len = x.dims()[0];
    const int& batch_size = x.dims()[1];
    const int& input_dim = x.dims()[2];
    const int& direction_num = is_bidirec ? 2 : 1;
    int in_dim_arr[in_out_dim_num] = {seq_len, batch_size, input_dim};
    int out_dim_arr[in_out_dim_num] = {
        seq_len, batch_size, direction_num * hidden_size};
    int proj_size = hidden_size;
    PADDLE_ENFORCE_EQ(
        num_layers,
        1,
        phi::errors::InvalidArgument(
            "MLU only support 1 num_layers, current num_layers is %s",
            num_layers));
    PADDLE_ENFORCE_EQ(
        init_h->dims()[0],
        num_layers * direction_num,
        phi::errors::InvalidArgument("The num_layers of in RNN layer must"
                                          " be the same as first dim of init"
                                          "hidden, but received num_layers:%d,"
                                          " dim:%d",
                                          num_layers,
                                          init_h->dims()[0]));
    PADDLE_ENFORCE_EQ(
        init_c->dims()[0],
        num_layers * direction_num,
        phi::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers,
            init_c->dims()[0]));

    std::vector<std::vector<std::pair<T*, size_t>>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(
        weight_list, num_layers, is_bidirec, &parameter_lists);

    for (unsigned int i = 0; i < weight_grad_list.size(); ++i) {
     dev_ctx.template Alloc<T>(weight_grad_list[i]);
    }
    std::vector<std::vector<std::pair<T*, size_t>>> parameter_lists_grad;
    parameter_lists_grad.resize(num_layers);
    reset_parameter_vector(
        weight_grad_list, num_layers, is_bidirec, &parameter_lists_grad);

    // allocate the memory and initization the input_grad
    x_grad->Resize(x.dims());
    dev_ctx.template Alloc<T>(x_grad);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0), x_grad);

    Tensor a, b;
    Tensor* dynamic_grad_pre_h = &a;
    Tensor* dynamic_grad_pre_c = &b;
    if (init_h_grad) {
      init_h_grad->Resize(last_h_grad->dims());
      dev_ctx.template Alloc<T>(init_h_grad);
      FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0), init_h_grad);
    } else {
      dynamic_grad_pre_h->Resize(last_h_grad->dims());
      dev_ctx.template Alloc<T>(dynamic_grad_pre_h);
      FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0), dynamic_grad_pre_h);
      init_h_grad = dynamic_grad_pre_h;
    }
    if (init_c_grad) {
      init_c_grad->Resize(last_c_grad->dims());
      dev_ctx.template Alloc<T>(init_c_grad);
    } else {
      dynamic_grad_pre_c->Resize(last_h_grad->dims());
      dev_ctx.template Alloc<T>(dynamic_grad_pre_c);
      init_c_grad = dynamic_grad_pre_c;
    }

    std::vector<int> seq_len_vec(batch_size, seq_len);
    if (sequence_length.is_initialized()) {  
     custom_kernel::TensorToVector(dev_ctx, *sequence_length, dev_ctx, &seq_len_vec);
    }
    cnnlDirectionMode_t direction =
        is_bidirec ? CNNL_RNN_BIDIRECTIONAL : CNNL_RNN_UNIDIRECTIONAL;

    MLUSeqDataDesc input_seq_data_desc(CNNL_SEQDATA_TNC,
                                       ToCnnlDataType(x.dtype()),
                                       in_out_dim_num,
                                       in_dim_arr,
                                       static_cast<int>(seq_len_vec.size()),
                                       seq_len_vec.data(),
                                       nullptr);
    MLUSeqDataDesc out_seq_data_desc(CNNL_SEQDATA_TNC,
                                     ToCnnlDataType(x.dtype()),
                                     in_out_dim_num,
                                     out_dim_arr,
                                     static_cast<int>(seq_len_vec.size()),
                                     seq_len_vec.data(),
                                     nullptr);
    MLUCnnlTensorDesc hx_desc(*init_h);
    MLUCnnlTensorDesc cx_desc(*init_c);
    MLURNNDesc rnn_desc(CNNL_LSTM,
                        CNNL_RNN_DOUBLE_BIAS,
                        direction,
                        CNNL_RNN_LINEAR_INPUT,
                        ToCnnlDataType(x.dtype()),
                        ToCnnlDataType(x.dtype()),
                        input_dim,
                        hidden_size,
                        /*projection*/ proj_size,
                        num_layers,
                        nullptr,
                        CNNL_RNN_PADDED_IO_DISABLED);
    rnn_desc.SetRNNMaskMode(CNNL_LSTM_MASK_ENABLED);

    // copy weight
    size_t weightspace_size;
    Tensor weightspace, dweightspace;
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetRNNWeightSpaceSize(
        GetHandleFromCTX(dev_ctx), rnn_desc.get(), &weightspace_size));

    weightspace.Resize({static_cast<int64_t>(weightspace_size)});
    dev_ctx.template Alloc<T>(&weightspace);
    dweightspace.Resize({static_cast<int64_t>(weightspace_size)});
    dev_ctx.template Alloc<T>(&dweightspace);
    void* weightspace_ptr = dev_ctx.template Alloc<T>(&weightspace);
    auto w_x = parameter_lists[0][0];
    auto w_h = parameter_lists[0][1];
    auto b_x = parameter_lists[0][2];
    auto b_h = parameter_lists[0][3];
    auto actual_total_w_size =
        w_x.second + w_h.second + b_x.second + b_h.second;

    void* w_x_ptr = weightspace_ptr;
    void* w_h_ptr = static_cast<char*>(weightspace_ptr) + w_x.second;
    void* b_x_ptr =
        static_cast<char*>(weightspace_ptr) + w_x.second + w_h.second;
    void* b_h_ptr = static_cast<char*>(weightspace_ptr) + w_x.second +
                    w_h.second + b_x.second;

    AsyncMemCpyD2D(nullptr,
          stream,
          w_x_ptr,
          w_x.first,
          w_x.second);

    AsyncMemCpyD2D(nullptr,
          stream,
          w_h_ptr,
          w_h.first,
          w_h.second);
    AsyncMemCpyD2D(nullptr,
          stream,
          b_x_ptr,
          b_x.first,
          b_x.second);
    AsyncMemCpyD2D(nullptr,
          stream,
          b_h_ptr,
          b_h.first,
          b_h.second);
    dev_ctx.Wait();

    if (is_bidirec) {
      auto bw_x = parameter_lists[0][4];
      auto bw_h = parameter_lists[0][5];
      auto bb_x = parameter_lists[0][6];
      auto bb_h = parameter_lists[0][7];
      void* bw_x_ptr =
          static_cast<char*>(weightspace_ptr) + actual_total_w_size;
      void* bw_h_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second;
      void* bb_x_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second + bw_h.second;
      void* bb_h_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second + bw_h.second +
                       bb_x.second;
      actual_total_w_size +=
          bw_x.second + bw_h.second + bb_x.second + bb_h.second;

    AsyncMemCpyD2D(nullptr,stream,
          bw_x_ptr,
          bw_x.first,
          bw_x.second);
    AsyncMemCpyD2D(nullptr,stream,
          bw_h_ptr,
          bw_h.first,
          bw_h.second);
    AsyncMemCpyD2D(nullptr,stream,
          bb_x_ptr,
          bb_x.first,
          bb_x.second);
    AsyncMemCpyD2D(nullptr,stream,
          bb_h_ptr,
          bb_h.first,
          bb_h.second);
    }
    dev_ctx.Wait();

    PADDLE_ENFORCE_EQ(weightspace_size,
                      actual_total_w_size,
                      phi::errors::InvalidArgument(
                          "The weightsize doesn't match"
                          " weightspace_size:%d, actual_total_w_size:%d",
                          weightspace_size,
                          actual_total_w_size));

    MLUCnnl::RNNBackward(dev_ctx,
                         rnn_desc.get(),
                         CNNL_WGRAD_MODE_SET,
                         seq_len_vec.data(),
                         GetBasePtr(&weightspace),
                         GetBasePtr(&dweightspace),
                         weightspace.numel() * sizeof(T),
                         input_seq_data_desc.get(),
                         GetBasePtr(&x),
                         GetBasePtr(x_grad),
                         out_seq_data_desc.get(),
                         GetBasePtr(&out),
                         GetBasePtr(&out_grad),
                         hx_desc.get(),
                         GetBasePtr(init_h),
                         GetBasePtr(last_h_grad),
                         GetBasePtr(init_h_grad),
                         cx_desc.get(),
                         GetBasePtr(init_c),
                         GetBasePtr(last_c_grad),
                         GetBasePtr(init_c_grad),
                         const_cast<void*>(GetBasePtr(&reserve)),
                         reserve.numel() * sizeof(T));

    void* dweightspace_ptr =dev_ctx.template Alloc<T>(&dweightspace);
    auto dw_x = parameter_lists_grad[0][0];
    auto dw_h = parameter_lists_grad[0][1];
    auto db_x = parameter_lists_grad[0][2];
    auto db_h = parameter_lists_grad[0][3];
    auto dactual_total_w_size =
        dw_x.second + dw_h.second + db_x.second + db_h.second;

    void* dw_x_ptr = dweightspace_ptr;
    void* dw_h_ptr = static_cast<char*>(dweightspace_ptr) + dw_x.second;
    void* db_x_ptr =
        static_cast<char*>(dweightspace_ptr) + dw_x.second + dw_h.second;
    void* db_h_ptr = static_cast<char*>(dweightspace_ptr) + dw_x.second +
                     dw_h.second + db_x.second;

    AsyncMemCpyD2D(nullptr,stream,
          dw_x.first,
          dw_x_ptr,
          dw_x.second);
    AsyncMemCpyD2D(nullptr,stream,
          dw_h.first,
          dw_h_ptr,
          dw_h.second);
    AsyncMemCpyD2D(nullptr,stream, 
          db_x.first,
          db_x_ptr,
          db_x.second);
    AsyncMemCpyD2D(nullptr,stream, 
          db_h.first,
          db_h_ptr,
          db_h.second);
    dev_ctx.Wait();

    if (is_bidirec) {
      auto dbw_x = parameter_lists_grad[0][4];
      auto dbw_h = parameter_lists_grad[0][5];
      auto dbb_x = parameter_lists_grad[0][6];
      auto dbb_h = parameter_lists_grad[0][7];
      void* dbw_x_ptr =
          static_cast<char*>(dweightspace_ptr) + dactual_total_w_size;
      void* dbw_h_ptr = static_cast<char*>(dweightspace_ptr) +
                        dactual_total_w_size + dbw_x.second;
      void* dbb_x_ptr = static_cast<char*>(dweightspace_ptr) +
                        dactual_total_w_size + dbw_x.second + dbw_h.second;
      void* dbb_h_ptr = static_cast<char*>(dweightspace_ptr) +
                        dactual_total_w_size + dbw_x.second + dbw_h.second +
                        dbb_x.second;
      dactual_total_w_size +=
          dbw_x.second + dbw_h.second + dbb_x.second + dbb_h.second;


    AsyncMemCpyD2D(nullptr,stream,
          dbw_x.first,
          dbw_x_ptr,
          dbw_x.second);
    AsyncMemCpyD2D(nullptr,stream,
          dbw_h.first,
          dbw_h_ptr,
          dbw_h.second);
    AsyncMemCpyD2D(nullptr,stream,
          dbb_x.first,
          dbb_x_ptr,
          dbb_x.second);
    AsyncMemCpyD2D(nullptr,stream,
          dbb_h.first,
          dbb_h_ptr,
          dbb_h.second);
    
    }
    dev_ctx.Wait();    

    PADDLE_ENFORCE_EQ(weightspace_size,
                      dactual_total_w_size,
                      phi::errors::InvalidArgument(
                          "The weightsize doesn't match"
                          " weightspace_size:%d, dactual_total_w_size:%d",
                          weightspace_size,
                          dactual_total_w_size));

}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(rnn,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::RnnKernel,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(rnn_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::RnnGradKernel,
                          float) {}

