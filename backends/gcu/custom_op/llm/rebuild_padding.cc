// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "custom_op/custom_op_common.h"

void DoRebuildPadding(float *output_data,
                      const float *input_data,
                      const int *cum_offset,
                      const int *seq_len_this_time,
                      const int *seq_len_decoder,
                      const int *seq_len_encoder,
                      const int *output_padding_offset,
                      const int max_input_length,
                      const int hidden_dim,
                      const int output_elem_nums) {
  for (int i = 0; i < output_elem_nums; ++i) {
    const int out_token_id = i / hidden_dim;
    const int bias_idx = i % hidden_dim;
    int bi = 0;
    int ori_token_id = 0;
    if (output_padding_offset != nullptr) {
      ori_token_id = out_token_id + output_padding_offset[out_token_id];
      bi = ori_token_id / max_input_length;
    } else {
      bi = out_token_id;
    }

    int seq_id = 0;
    if (seq_len_this_time[bi] == 0) continue;
    if (seq_len_decoder[bi] == 0 && seq_len_encoder[bi] == 0) continue;
    // if encoder, get last token; just decoder, get first token.
    if (seq_len_encoder[bi] > 0) seq_id = seq_len_encoder[bi] - 1;

    int input_token_id = 0;
    if (output_padding_offset != nullptr) {
      input_token_id = ori_token_id - cum_offset[bi] + seq_id;
    } else {
      input_token_id = bi * max_input_length - cum_offset[bi] + seq_id;
    }
    int src_offset = input_token_id * hidden_dim + bias_idx;
    output_data[i] = input_data[src_offset];
  }
}

std::vector<paddle::Tensor> RebuildPadding(
    const paddle::Tensor &tmp_out,            // [token_num, hidden_dim]
    const paddle::Tensor &cum_offsets,        // [bsz]
    const paddle::Tensor &seq_len_this_time,  // [bsz]
    const paddle::Tensor &seq_lens_decoder,   // [bsz, 1]
    const paddle::Tensor &seq_lens_encoder,   // [bsz, 1]
    const paddle::optional<paddle::Tensor>
        &output_padding_offset,  // [token_num]
    int max_input_length) {
  PADDLE_GCU_KERNEL_TRACE("rebuild_padding_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: rebuild_padding_gcu";
  std::vector<int64_t> tmp_out_shape = tmp_out.shape();
  const int token_num = tmp_out_shape[0];
  const int hidden_dim = tmp_out_shape[1];
  const int bsz = cum_offsets.shape()[0];
  auto tmp_out_f32 =
      paddle::experimental::cast(tmp_out, phi::DataType::FLOAT32);

  auto tmp_out_f32_cpu = tmp_out_f32.copy_to(paddle::CPUPlace(), false);
  auto cum_offsets_cpu = cum_offsets.copy_to(paddle::CPUPlace(), false);
  auto seq_len_this_time_cpu =
      seq_len_this_time.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_encoder_cpu =
      seq_lens_encoder.copy_to(paddle::CPUPlace(), true);

  paddle::Tensor out_f32_cpu;
  const int *output_padding_offset_ptr = nullptr;
  if (output_padding_offset) {
    int need_delete_token_num = 0;
    const int *seq_lens_encoder_ptr = seq_lens_encoder_cpu.data<int>();
    for (int i = 0; i < bsz; ++i) {
      if (seq_lens_encoder_ptr[i] > 0) {
        need_delete_token_num += seq_lens_encoder_ptr[i] - 1;
      }
    }
    out_f32_cpu = paddle::full({token_num - need_delete_token_num, hidden_dim},
                               0,
                               phi::DataType::FLOAT32,
                               paddle::CPUPlace());
    output_padding_offset_ptr = output_padding_offset.get_ptr()->data<int>();
  } else {
    out_f32_cpu = paddle::full(
        {bsz, hidden_dim}, 0, phi::DataType::FLOAT32, paddle::CPUPlace());
  }

  int output_elem_nums = out_f32_cpu.numel();
  DoRebuildPadding(out_f32_cpu.data<float>(),
                   tmp_out_f32_cpu.data<float>(),
                   cum_offsets_cpu.data<int>(),
                   seq_len_this_time_cpu.data<int>(),
                   seq_lens_decoder_cpu.data<int>(),
                   seq_lens_encoder_cpu.data<int>(),
                   output_padding_offset_ptr,
                   max_input_length,
                   hidden_dim,
                   output_elem_nums);

  auto out_f32 = out_f32_cpu.copy_to(tmp_out.place(), true);
  auto out = paddle::experimental::cast(out_f32, tmp_out.dtype());
  return {out};
}

std::vector<std::vector<int64_t>> RebuildPaddingInferShape(
    const std::vector<int64_t> &tmp_out_shape,
    const std::vector<int64_t> &cum_offsets_shape,
    const std::vector<int64_t> &seq_len_this_time_shape,
    const std::vector<int64_t> &seq_lens_decoder_shape,
    const std::vector<int64_t> &seq_lens_encoder_shape,
    const paddle::optional<std::vector<int64_t>> &output_padding_offset_shape) {
  int64_t dim_embed = tmp_out_shape[1];
  // whether speculative decoding
  if (output_padding_offset_shape) {
    return {{-1, dim_embed}};
  } else {
    int64_t bsz = cum_offsets_shape[0];
    return {{bsz, dim_embed}};
  }
}

std::vector<paddle::DataType> RebuildPaddingInferDtype(
    const paddle::DataType &tmp_out_dtype,
    const paddle::DataType &cum_offsets_dtype,
    const paddle::DataType &seq_len_this_time_dtype,
    const paddle::DataType &seq_lens_decoder_dtype,
    const paddle::DataType &seq_lens_encoder_dtype,
    const paddle::optional<paddle::DataType> &output_padding_offset_dtype) {
  return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding_gcu)
    .Inputs({"tmp_out",
             "cum_offsets",
             "seq_len_this_time",
             "seq_lens_decoder",
             "seq_lens_encoder",
             paddle::Optional("output_padding_offset")})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(RebuildPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingInferDtype));
