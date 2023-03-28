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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::vector<int>& axis,
                phi::DenseTensor* out);

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out);

template <typename T, typename Context>
void ResetParameterVector(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& raw_params_vec,
    int num_layers,
    bool is_bidirec,
    std::vector<std::vector<phi::DenseTensor>>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    std::vector<phi::DenseTensor> tensor_list;
    tensor_list.reserve(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      phi::DenseTensor tensor_holder;
      tensor_list.emplace_back(tensor_holder);
    }
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      tensor_list[j] = *raw_params_vec[tensor_idx];
    }
    params_vec->emplace_back(tensor_list);
  }
}

template <typename T, typename Context>
void SlicePreState(const Context& dev_ctx,
                   const phi::DenseTensor& pre_state,
                   int num_layers,
                   bool is_bidirec,
                   std::vector<std::vector<phi::DenseTensor>>* pre_state_vec) {
  auto stream = dev_ctx.stream();
  const int& direction_num = is_bidirec ? 2 : 1;
  std::vector<int64_t> size_vec = phi::vectorize(pre_state.dims());
  size_vec[0] = 1;
  std::vector<int64_t> start_vec(size_vec.size());

  for (int i = 0; i < num_layers; ++i) {
    std::vector<phi::DenseTensor> tensor_list;
    for (int j = 0; j < direction_num; ++j) {
      std::vector<int64_t> start_vec_tmp = start_vec;
      std::vector<int64_t> size_vec_tmp = size_vec;
      phi::DenseTensor tensor_slice;
      tensor_slice.Resize(phi::make_ddim(size_vec));
      dev_ctx.template Alloc<T>(&tensor_slice);
      NpuOpRunner slice_runner;
      slice_runner.SetType("Slice")
          .AddInput(pre_state)
          .AddInput(dev_ctx, std::move(start_vec_tmp))
          .AddInput(dev_ctx, std::move(size_vec_tmp))
          .AddOutput(tensor_slice);
      slice_runner.Run(stream);
      tensor_list.emplace_back(tensor_slice);
      start_vec[0]++;
    }
    pre_state_vec->emplace_back(tensor_list);
  }
}

template <typename T, typename Context>
void LSTMKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& w_i,
                const phi::DenseTensor& w_h,
                const phi::DenseTensor& b,
                const phi::DenseTensor& init_h,
                const phi::DenseTensor& init_c,
                const std::vector<int>& SequenceLength,
                float dropout_prob,
                phi::DenseTensor* out,
                phi::DenseTensor* last_h,
                phi::DenseTensor* last_c) {
  auto stream = dev_ctx.stream();
  phi::DenseTensor seq;
  seq.Resize(phi::make_ddim({SequenceLength.size()}));
  dev_ctx.template Alloc<int>(&seq);
  TensorFromVector(dev_ctx, SequenceLength, dev_ctx, &seq);

  phi::DenseTensor w;
  w.Resize(phi::make_ddim({w_i.dims()[0], w_h.dims()[1] + w_i.dims()[1]}));
  dev_ctx.template Alloc<T>(&w);
  std::vector<std::string> names;
  names.emplace_back("concat_dim");
  for (size_t i = 0; i < 2; ++i) {
    names.emplace_back("x" + std::to_string(i));
  }
  NpuOpRunner concat_runner;
  concat_runner.SetType("Concat")
      .AddInput(dev_ctx, std::vector<int>(1, 1))
      .AddInputs({w_i, w_h})
      .AddOutput(w)
      .AddAttr("N", static_cast<int>(2))
      .AddInputNames(names);
  concat_runner.Run(stream);

  phi::DenseTensor weight;
  weight.Resize(phi::make_ddim({w.dims()[1], w.dims()[0]}));
  custom_kernel::TransposeKernel<T, Context>(dev_ctx, w, {1, 0}, &weight);

  phi::DenseTensor i, j, f, o, tanhc;
  i.Resize(out->dims());
  j.Resize(out->dims());
  f.Resize(out->dims());
  o.Resize(out->dims());
  tanhc.Resize(out->dims());
  dev_ctx.template Alloc<T>(&i);
  dev_ctx.template Alloc<T>(&j);
  dev_ctx.template Alloc<T>(&f);
  dev_ctx.template Alloc<T>(&o);
  dev_ctx.template Alloc<T>(&tanhc);

  NpuOpRunner runner;
  runner.SetType("DynamicRNN")
      .AddInput(x)
      .AddInput(weight)
      .AddInput(b)
      .AddInput(seq)
      .AddInput(init_h)
      .AddInput(init_c)
      .AddOutput(*out)
      .AddOutput(*last_h)
      .AddOutput(*last_c)
      .AddOutput(i)
      .AddOutput(j)
      .AddOutput(f)
      .AddOutput(o)
      .AddOutput(tanhc)
      .AddAttr("cell_type", std::string("LSTM"))
      .AddAttr("direction", std::string("UNIDIRECTIONAL"))
      .AddAttr("cell_depth", 1)
      .AddAttr("use_peephole", false)
      .AddAttr("keep_prob", static_cast<float>(1 - dropout_prob))
      .AddAttr("cell_clip", static_cast<float>(-1.0))
      .AddAttr("num_proj", static_cast<int64_t>(0))
      .AddAttr("time_major", true)
      .AddAttr("activation", std::string("tanh"))
      .AddAttr("forget_bias", static_cast<float>(0.0))
      .AddAttr("gate_order", std::string("ifjo"))
      .AddAttr("is_training", true);
  runner.Run(stream);
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
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(out);

  // get dropout_state
  dropout_state->Resize(out->dims());
  dev_ctx.template Alloc<T>(dropout_state);
  FillNpuTensorWithConstant<uint8_t>(
      dropout_state, dev_ctx, static_cast<uint8_t>(1));
  dropout_state->Resize(out->dims());

  // prestate and output state
  auto init_h = pre_state[0];
  auto init_c = pre_state[1];
  auto last_h = state[0];
  auto last_c = state[1];

  // check shape
  int seq_length = x.dims()[0];  // time_step
  int batch_size = x.dims()[1];
  int input_size_local = x.dims()[2];
  const int& direction_num = is_bidirec ? 2 : 1;
  int gate_num = 4;

  dev_ctx.template Alloc<T>(last_h);

  const auto& init_h_dims = init_h->dims();
  PADDLE_ENFORCE_EQ(init_h_dims[0],
                    num_layers * direction_num,
                    phi::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of init hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers,
                        init_h_dims[0]));

  // reset sequence_length
  std::vector<int> SequenceLength(batch_size, seq_length);
  bool has_seq_length = sequence_length.is_initialized();
  if (has_seq_length) {
    TensorToVector(
        dev_ctx, *sequence_length.get_ptr(), dev_ctx, &SequenceLength);
  }

  // reset parameter and init_h
  std::vector<std::vector<phi::DenseTensor>> parameter_lists, init_h_list;
  parameter_lists.reserve(num_layers);
  init_h_list.reserve(num_layers);
  custom_kernel::ResetParameterVector<T, Context>(
      dev_ctx, weight_list, num_layers, is_bidirec, &parameter_lists);
  custom_kernel::SlicePreState<T, Context>(
      dev_ctx, *init_h, num_layers, is_bidirec, &init_h_list);

  if (mode == "LSTM") {
    const auto& init_c_dims = init_c->dims();
    PADDLE_ENFORCE_EQ(init_c_dims[0],
                      num_layers * direction_num,
                      phi::errors::InvalidArgument(
                          "The num_layers of in RNN layer must be the same as "
                          "first dim of cell state hidden, but received"
                          " num_layers:%d, dim:%d",
                          num_layers,
                          init_c_dims[0]));

    // reset init_c
    std::vector<std::vector<phi::DenseTensor>> init_c_list;
    init_c_list.reserve(num_layers);
    custom_kernel::SlicePreState<T, Context>(
        dev_ctx, *init_c, num_layers, is_bidirec, &init_c_list);

    // alloc output
    int hidden_data_idx = (num_layers - 1);
    hidden_data_idx += (gate_num + 2) * num_layers;
    const int& block_size =
        direction_num * seq_length * batch_size * hidden_size;
    reserve->Resize({hidden_data_idx, block_size});
    dev_ctx.template Alloc<T>(reserve);
    dev_ctx.template Alloc<T>(last_c);
    std::vector<phi::DenseTensor> out_vec, last_h_vec, last_c_vec;

    phi::DenseTensor input(x);

    for (size_t i = 0; i < num_layers; ++i) {
      out_vec.clear();
      phi::DenseTensor out_tmp_f, last_h_tmp_f, last_c_tmp_f, last_h_slice_f,
          last_c_slice_f, bias_f;
      out_tmp_f.Resize(phi::make_ddim({seq_length, batch_size, hidden_size}));
      last_h_tmp_f.Resize(
          phi::make_ddim({seq_length, batch_size, hidden_size}));
      last_c_tmp_f.Resize(
          phi::make_ddim({seq_length, batch_size, hidden_size}));
      last_h_slice_f.Resize(phi::make_ddim({1, batch_size, hidden_size}));
      last_c_slice_f.Resize(phi::make_ddim({1, batch_size, hidden_size}));
      bias_f.Resize(parameter_lists[i][2].dims());
      dev_ctx.template Alloc<T>(&out_tmp_f);
      dev_ctx.template Alloc<T>(&last_h_tmp_f);
      dev_ctx.template Alloc<T>(&last_c_tmp_f);
      dev_ctx.template Alloc<T>(&last_h_slice_f);
      dev_ctx.template Alloc<T>(&last_c_slice_f);
      dev_ctx.template Alloc<T>(&bias_f);

      if (!is_bidirec) {
        // get input bias
        const auto& add_runner =
            NpuOpRunner("Add",
                        {parameter_lists[i][2], parameter_lists[i][3]},
                        {bias_f},
                        {});
        add_runner.Run(stream);

        custom_kernel::LSTMKernel<T, Context>(dev_ctx,
                                              input,
                                              parameter_lists[i][0],
                                              parameter_lists[i][1],
                                              bias_f,
                                              init_h_list[i][0],
                                              init_c_list[i][0],
                                              SequenceLength,
                                              dropout_prob,
                                              &out_tmp_f,
                                              &last_h_tmp_f,
                                              &last_c_tmp_f);
        out_vec.emplace_back(out_tmp_f);

        NpuOpRunner slice_runner1;
        slice_runner1.SetType("Slice")
            .AddInput(last_h_tmp_f)
            .AddInput(dev_ctx, std::vector<int64_t>{seq_length - 1, 0, 0})
            .AddInput(dev_ctx, std::vector<int64_t>{1, batch_size, hidden_size})
            .AddOutput(last_h_slice_f);
        slice_runner1.Run(stream);

        NpuOpRunner slice_runner2;
        slice_runner2.SetType("Slice")
            .AddInput(last_c_tmp_f)
            .AddInput(dev_ctx, std::vector<int64_t>{seq_length - 1, 0, 0})
            .AddInput(dev_ctx, std::vector<int64_t>{1, batch_size, hidden_size})
            .AddOutput(last_c_slice_f);
        slice_runner2.Run(stream);

        last_h_vec.emplace_back(last_h_slice_f);
        last_c_vec.emplace_back(last_c_slice_f);
      } else {
        phi::DenseTensor out_tmp_b, last_h_tmp_b, last_c_tmp_b, last_h_slice_b,
            last_c_slice_b, bias_b, input_flip, out_tmp_b_flip,
            last_h_tmp_b_flip, last_c_tmp_b_flip;
        out_tmp_b.Resize(phi::make_ddim({seq_length, batch_size, hidden_size}));
        last_h_tmp_b.Resize(
            phi::make_ddim({seq_length, batch_size, hidden_size}));
        last_c_tmp_b.Resize(
            phi::make_ddim({seq_length, batch_size, hidden_size}));
        last_h_tmp_b_flip.Resize(
            phi::make_ddim({seq_length, batch_size, hidden_size}));
        last_c_tmp_b_flip.Resize(
            phi::make_ddim({seq_length, batch_size, hidden_size}));
        last_h_slice_b.Resize(phi::make_ddim({1, batch_size, hidden_size}));
        last_c_slice_b.Resize(phi::make_ddim({1, batch_size, hidden_size}));
        bias_b.Resize(parameter_lists[i][6].dims());
        input_flip.Resize(input.dims());
        out_tmp_b_flip.Resize({seq_length, batch_size, hidden_size});
        dev_ctx.template Alloc<T>(&out_tmp_b);
        dev_ctx.template Alloc<T>(&last_h_tmp_b);
        dev_ctx.template Alloc<T>(&last_c_tmp_b);
        dev_ctx.template Alloc<T>(&last_h_tmp_b_flip);
        dev_ctx.template Alloc<T>(&last_c_tmp_b_flip);
        dev_ctx.template Alloc<T>(&last_h_slice_b);
        dev_ctx.template Alloc<T>(&last_c_slice_b);
        dev_ctx.template Alloc<T>(&bias_b);
        dev_ctx.template Alloc<T>(&input_flip);
        dev_ctx.template Alloc<T>(&out_tmp_b_flip);

        const auto& add_runner1 =
            NpuOpRunner("Add",
                        {parameter_lists[i][2], parameter_lists[i][3]},
                        {bias_f},
                        {});
        add_runner1.Run(stream);

        const auto& add_runner2 =
            NpuOpRunner("Add",
                        {parameter_lists[i][6], parameter_lists[i][7]},
                        {bias_b},
                        {});
        add_runner2.Run(stream);

        // forward direction
        custom_kernel::LSTMKernel<T, Context>(dev_ctx,
                                              input,
                                              parameter_lists[i][0],
                                              parameter_lists[i][1],
                                              bias_f,
                                              init_h_list[i][0],
                                              init_c_list[i][0],
                                              SequenceLength,
                                              dropout_prob,
                                              &out_tmp_f,
                                              &last_h_tmp_f,
                                              &last_c_tmp_f);
        out_vec.emplace_back(out_tmp_f);

        NpuOpRunner slice_runner1;
        slice_runner1.SetType("Slice")
            .AddInput(last_h_tmp_f)
            .AddInput(dev_ctx, std::vector<int64_t>{seq_length - 1, 0, 0})
            .AddInput(dev_ctx, std::vector<int64_t>{1, batch_size, hidden_size})
            .AddOutput(last_h_slice_f);
        slice_runner1.Run(stream);

        NpuOpRunner slice_runner2;
        slice_runner2.SetType("Slice")
            .AddInput(last_c_tmp_f)
            .AddInput(dev_ctx, std::vector<int64_t>{seq_length - 1, 0, 0})
            .AddInput(dev_ctx, std::vector<int64_t>{1, batch_size, hidden_size})
            .AddOutput(last_c_slice_f);
        slice_runner2.Run(stream);

        last_h_vec.emplace_back(last_h_slice_f);
        last_c_vec.emplace_back(last_c_slice_f);

        // flip input
        custom_kernel::FlipKernel<T, Context>(dev_ctx, input, {0}, &input_flip);
        // backward direction
        custom_kernel::LSTMKernel<T, Context>(dev_ctx,
                                              input_flip,
                                              parameter_lists[i][4],
                                              parameter_lists[i][5],
                                              bias_b,
                                              init_h_list[i][1],
                                              init_c_list[i][1],
                                              SequenceLength,
                                              dropout_prob,
                                              &out_tmp_b,
                                              &last_h_tmp_b,
                                              &last_c_tmp_b);
        // flip output
        custom_kernel::FlipKernel<T, Context>(
            dev_ctx, out_tmp_b, {0}, &out_tmp_b_flip);
        custom_kernel::FlipKernel<T, Context>(
            dev_ctx, last_h_tmp_b, {0}, &last_h_tmp_b_flip);
        custom_kernel::FlipKernel<T, Context>(
            dev_ctx, last_c_tmp_b, {0}, &last_c_tmp_b_flip);
        out_vec.emplace_back(out_tmp_b_flip);

        NpuOpRunner slice_runner3;
        slice_runner3.SetType("Slice")
            .AddInput(last_h_tmp_b_flip)
            .AddInput(dev_ctx, std::vector<int64_t>{0, 0, 0})
            .AddInput(dev_ctx, std::vector<int64_t>{1, batch_size, hidden_size})
            .AddOutput(last_h_slice_b);
        slice_runner3.Run(stream);

        NpuOpRunner slice_runner4;
        slice_runner4.SetType("Slice")
            .AddInput(last_c_tmp_b_flip)
            .AddInput(dev_ctx, std::vector<int64_t>{0, 0, 0})
            .AddInput(dev_ctx, std::vector<int64_t>{1, batch_size, hidden_size})
            .AddOutput(last_c_slice_b);
        slice_runner4.Run(stream);

        last_h_vec.emplace_back(last_h_slice_b);
        last_c_vec.emplace_back(last_c_slice_b);
      }
      if (out_vec.size() > 1) {
        std::vector<std::string> names;
        names.emplace_back("concat_dim");
        for (size_t i = 0; i < out_vec.size(); ++i) {
          names.emplace_back("x" + std::to_string(i));
        }

        phi::DenseTensor out_concat;
        out_concat.Resize(
            phi::make_ddim({seq_length, batch_size, hidden_size * 2}));
        dev_ctx.template Alloc<T>(&out_concat);
        NpuOpRunner concat_runner;
        concat_runner.SetType("Concat")
            .AddInput(dev_ctx, std::vector<int>(1, 2))
            .AddInputs(out_vec)
            .AddOutput(out_concat)
            .AddAttr("N", static_cast<int>(out_vec.size()))
            .AddInputNames(names);
        concat_runner.Run(stream);
        input = out_concat;
      } else {
        input = out_vec[0];
      }
    }
    if (out_vec.size() > 1) {
      std::vector<std::string> names;
      names.emplace_back("concat_dim");
      for (size_t i = 0; i < out_vec.size(); ++i) {
        names.emplace_back("x" + std::to_string(i));
      }
      NpuOpRunner concat_runner;
      concat_runner.SetType("Concat")
          .AddInput(dev_ctx, std::vector<int>(1, 2))
          .AddInputs(out_vec)
          .AddOutput(*out)
          .AddAttr("N", static_cast<int>(out_vec.size()))
          .AddInputNames(names);
      concat_runner.Run(stream);
    } else {
      TensorCopy(dev_ctx, out_vec[0], true, out);
    }

    if (last_h_vec.size() > 1) {
      std::vector<std::string> names;
      names.emplace_back("concat_dim");
      for (size_t i = 0; i < last_h_vec.size(); ++i) {
        names.emplace_back("x" + std::to_string(i));
      }
      NpuOpRunner concat_runner;
      concat_runner.SetType("Concat")
          .AddInput(dev_ctx, std::vector<int>(1, 0))
          .AddInputs(last_h_vec)
          .AddOutput(*last_h)
          .AddAttr("N", static_cast<int>(out_vec.size()))
          .AddInputNames(names);
      concat_runner.Run(stream);
    } else {
      TensorCopy(dev_ctx, last_h_vec[0], true, last_h);
    }

    if (last_c_vec.size() > 1) {
      std::vector<std::string> names;
      names.emplace_back("concat_dim");
      for (size_t i = 0; i < last_c_vec.size(); ++i) {
        names.emplace_back("x" + std::to_string(i));
      }
      NpuOpRunner concat_runner;
      concat_runner.SetType("Concat")
          .AddInput(dev_ctx, std::vector<int>(1, 0))
          .AddInputs(last_c_vec)
          .AddOutput(*last_c)
          .AddAttr("N", static_cast<int>(out_vec.size()))
          .AddInputNames(names);
      concat_runner.Run(stream);
    } else {
      TensorCopy(dev_ctx, last_c_vec[0], true, last_c);
    }
  } else if (mode == "GRU") {
    PADDLE_ENFORCE_EQ(num_layers,
                      1,
                      phi::errors::InvalidArgument(
                          "NPU rnn kernel only support 1 layer for GRU mode at "
                          "present, but received num_layers=%d.",
                          num_layers));
    gate_num = 3;
    int hidden_data_idx = (num_layers - 1);
    hidden_data_idx += (gate_num + 1) * num_layers;
    const int& block_size =
        direction_num * seq_length * batch_size * hidden_size;
    reserve->Resize({hidden_data_idx, block_size});
    dev_ctx.template Alloc<T>(reserve);
    // TODO(songkai05): implement GRU mode
  } else {
    phi::errors::InvalidArgument(
        "Custom NPU only support LSTM and GRU mode now, current mode is: %s",
        mode);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    rnn, npu, ALL_LAYOUT, custom_kernel::RnnKernel, float) {}

// TODO(songkai05): implement grad op
// PD_REGISTER_PLUGIN_KERNEL(rnn_grad,
//                           npu,
//                           ALL_LAYOUT,
//                           custom_kernel::RnnGradKernel,
//                           float) {}
