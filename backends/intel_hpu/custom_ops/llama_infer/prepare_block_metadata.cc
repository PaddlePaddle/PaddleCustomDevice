// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

// retrun: amax and where(>0)
std::tuple<int, int, int, std::vector<int>> get_max_and_where_nonzero(
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    const int elem_cnt,
    int max_num_batched_tokens,
    int block_size) {
  int max_seq_len_without_context = 0;
  int max_context_len = 0;
  std::vector<int> valid_batch;
  for (int i = 0; i < elem_cnt; ++i) {
    if (seq_lens_encoder[i] > 0) {
      valid_batch.push_back(i);
      int candidate_max_seq_len_without_context = max_seq_len_without_context;
      if (seq_lens_encoder[i] > candidate_max_seq_len_without_context) {
        candidate_max_seq_len_without_context = seq_lens_encoder[i];
      }
      int current_num_batched_tokens =
          (candidate_max_seq_len_without_context + block_size - 1) /
          block_size * block_size * (valid_batch.size());
      if (current_num_batched_tokens > max_num_batched_tokens) {
        valid_batch.pop_back();
        break;
      }
      max_seq_len_without_context = candidate_max_seq_len_without_context;
      if (seq_lens_decoder[i] > max_context_len) {
        max_context_len = seq_lens_decoder[i];
      }
    }
  }
  int max_seq_len_with_context = max_seq_len_without_context + max_context_len;
  return {max_seq_len_without_context,
          max_seq_len_with_context,
          max_context_len,
          valid_batch};
}

// return: where(>0)
std::vector<int> where_nonzero(int* seq_lens_decoder, int elem_cnt) {
  std::vector<int> valid_batch;
  for (int i = 0; i < elem_cnt; ++i) {
    if (seq_lens_decoder[i] > 0) {
      valid_batch.push_back(i);
    }
  }
  return valid_batch;
}

int round_up(int value, int k) { return ((value + k - 1) / k) * k; }

int next_pow2(int value, int base) {
  int res = base;
  while (value > 1) {
    value = (value + 1) / 2;
    res *= 2;
  }
  return res;
}

int find_bucket(int value, const int bstep, const int max_batches) {
  const int bmin = 1;

  if (value <= bmin) {
    return bmin;
  } else {
    int next_step = round_up(value, bstep);
    int next_pow = next_pow2(value, bmin);
    return std::min(std::min(next_step, next_pow), max_batches);
  }
}

// in: input_p[B, input_linewidth]
// out: padded[padded_B, padded_linewidth]
// abstract input_p[B] based on valid_batches, fill in padded
template <typename T>
void pad_fill(const T* input_p,
              T* padded,
              int valid_batches_size,
              int input_linewidth,
              int padded_linewidth) {
  int copy_len = std::min(input_linewidth, padded_linewidth);
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int i = 0; i < valid_batches_size; ++i) {
    for (int j = 0; j < copy_len; ++j) {
      padded[i * padded_linewidth + j] = input_p[i * input_linewidth + j];
    }
  }
}

template <typename T>
void pad_fill(const T* input_p,
              const T* offsets,
              T* padded,
              std::vector<int> valid_batches,
              int input_linewidth,
              int padded_linewidth,
              int max_context_len,
              int block_size) {
  int copy_len = std::min(input_linewidth, padded_linewidth);
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int i = 0; i < static_cast<int>(valid_batches.size()); ++i) {
    for (int j = (max_context_len - offsets[valid_batches[i]]) / block_size,
             k = 0;
         j < copy_len && k < copy_len;
         ++j, ++k) {
      padded[i * padded_linewidth + j] =
          input_p[valid_batches[i] * input_linewidth + k];
    }
  }
}

template <typename T>
void pad_fill(const T* input_p,
              const T* offsets,
              T* padded,
              std::vector<int> valid_batches,
              int input_linewidth,
              int padded_linewidth,
              int block_size) {
  int copy_len = std::min(input_linewidth, padded_linewidth);
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int i = 0; i < static_cast<int>(valid_batches.size()); ++i) {
    for (int j = 0; j < copy_len; ++j) {
      padded[i * padded_linewidth + j] =
          input_p[valid_batches[i] * input_linewidth + j +
                  offsets[valid_batches[i]] / block_size];
    }
  }
}

template <typename T>
void pad_fill(const T* input_p, T* padded, std::vector<int> valid_batches) {
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int i = 0; i < static_cast<int>(valid_batches.size()); ++i) {
    padded[i] = input_p[valid_batches[i]];
  }
}

// in: seq_lens_decoder, block_tables
// out: block_indices, block_offset
// return last_block_pos, seq_lens
// abstract seq_lens_decoder[B] based on valid_batches
// get block based on seq_len-1
std::pair<std::vector<int32_t>, std::vector<int32_t>> index_sample(
    int32_t* seq_lens_decoder,
    int32_t* block_tables,
    int32_t* block_indices,
    int32_t* block_offset,
    std::vector<int> valid_batches,
    int max_blocks_each,
    int block_size) {
  std::vector<int32_t> last_block_pos, seq_lens;

  for (int i = 0; i < static_cast<int>(valid_batches.size()); ++i) {
    int seq_len = seq_lens_decoder[valid_batches[i]];
    int index = seq_len - 1;
    int block = valid_batches[i] * max_blocks_each + index / block_size;
    block_indices[i] = block_tables[block];
    block_offset[i] = index % block_size;
    last_block_pos.push_back((index / block_size) + 1);
    seq_lens.push_back(seq_len);
  }

  return {last_block_pos, seq_lens};
}

// return: block_list, block_groups
// get block number from block_tables based on valid_batches
// the amount is last_block_pos, record the same batch to block_groups
std::pair<std::vector<int>, std::vector<int>> get_block_list_groups(
    const std::vector<int>& valid_batches,
    int32_t* block_tables,
    const std::vector<int32_t>& last_block_pos,
    int max_blocks_each) {
  std::vector<int> block_list;
  std::vector<int> block_groups;

  for (size_t group_index = 0; group_index < valid_batches.size();
       ++group_index) {
    int valid_batch = valid_batches[group_index];
    for (int i = 0; i < last_block_pos[group_index]; ++i) {
      block_list.push_back(block_tables[valid_batch * max_blocks_each + i]);
      block_groups.push_back(group_index);
    }
  }

  return {block_list, block_groups};
}

std::vector<paddle::Tensor> PrepareBlockMetadata(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& rope_emb,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    int block_size,
    std::string dtype,
    int max_num_batched_tokens) {
  auto hpu_place = rope_emb.place();
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(hpu_place));
  auto block_tables_cpu = block_tables.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_encoder_cpu =
      seq_lens_encoder.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), true);

  const char* env_prefill_batch_step = std::getenv("BATCH_STEP_PREFILL");
  const int batch_step_prefill =
      env_prefill_batch_step ? std::atoi(env_prefill_batch_step) : 1;
  const char* env_context_block_step =
      std::getenv("CONTEXT_BLOCK_STEP_PREFILL");
  const int context_block_step_prefill =
      env_context_block_step ? std::atoi(env_context_block_step) : 1;
  const char* env_decode_batch_step = std::getenv("BATCH_STEP_DECODE");
  const int batch_step_decode =
      env_decode_batch_step ? std::atoi(env_decode_batch_step) : 4;
  const char* env_block_step = std::getenv("BLOCK_STEP_DECODE");
  const int block_step = env_block_step ? std::atoi(env_block_step) : 16;

  const int max_batches_in = input_ids.shape()[0];
  const int max_seq_len = input_ids.shape()[1];
  const int max_blocks_each = block_tables.shape()[1];
  phi::DataType device_dtype = phi::StringToDataType(dtype);

  auto [max_enc_len_without_context,
        max_enc_len_with_context,
        max_context_len,
        valid_batches_enc] =
      get_max_and_where_nonzero(
          const_cast<int*>(seq_lens_encoder_cpu.data<int>()),
          const_cast<int*>(seq_lens_decoder_cpu.data<int>()),
          max_batches_in,
          max_num_batched_tokens,
          block_size);
  int enc_count = valid_batches_enc.size();

  auto valid_batches_dec = where_nonzero(
      const_cast<int*>(seq_lens_decoder_cpu.data<int>()), max_batches_in);
  int dec_count = valid_batches_dec.size();

  auto dummy_tensor =
      paddle::full({1}, 0, phi::DataType::FLOAT32, paddle::CPUPlace());

  if (enc_count > 0) {
    int total_batch = enc_count;
    auto valid_batches_tensor =
        paddle::full({static_cast<int64_t>(valid_batches_enc.size())},
                     0,
                     paddle::DataType::INT32,
                     paddle::CPUPlace());
    std::memcpy(valid_batches_tensor.data<int>(),
                valid_batches_enc.data(),
                valid_batches_enc.size() * sizeof(int));
    auto batch_ids = custom_kernel::copy_tensor_wrapper(
        dev_ctx, valid_batches_tensor, hpu_place);

    auto input_ids_selected =
        paddle::experimental::index_select(input_ids, batch_ids, {0});

    auto input_ids_cpu = input_ids_selected.copy_to(paddle::CPUPlace(), true);

    int max_buckets_without_context =
        (max_enc_len_without_context + block_size - 1) / block_size;
    int max_prompt_len_without_context =
        max_buckets_without_context * block_size;

    auto src_padded =
        paddle::full({total_batch * max_prompt_len_without_context},
                     0,
                     paddle::DataType::INT64,
                     paddle::CPUPlace());
    pad_fill<int64_t>(const_cast<int64_t*>(input_ids_cpu.data<int64_t>()),
                      reinterpret_cast<int64_t*>(src_padded.data<int64_t>()),
                      static_cast<int>(valid_batches_enc.size()),
                      max_seq_len,
                      max_prompt_len_without_context);

    auto blk_padded = paddle::full({total_batch * max_buckets_without_context},
                                   -1,
                                   paddle::DataType::INT32,
                                   paddle::CPUPlace());
    pad_fill<int32_t>(
        const_cast<int32_t*>(block_tables_cpu.data<int32_t>()),
        const_cast<int32_t*>(seq_lens_decoder_cpu.data<int32_t>()),
        reinterpret_cast<int32_t*>(blk_padded.data<int32_t>()),
        valid_batches_enc,
        max_blocks_each,
        max_buckets_without_context,
        block_size);

    auto blk_padded_hpu =
        custom_kernel::copy_tensor_wrapper(dev_ctx, blk_padded, hpu_place);

    int max_buckets_with_context =
        (max_enc_len_with_context + block_size - 1) / block_size;
    max_buckets_with_context =
        ((max_buckets_with_context + context_block_step_prefill - 1) /
         context_block_step_prefill) *
        context_block_step_prefill;
    int max_prompt_len_with_context = max_buckets_with_context * block_size;

    auto block_list_padded =
        paddle::full({total_batch * max_buckets_with_context},
                     -1,
                     paddle::DataType::INT32,
                     paddle::CPUPlace());
    pad_fill<int32_t>(
        const_cast<int32_t*>(block_tables_cpu.data<int32_t>()),
        const_cast<int32_t*>(seq_lens_decoder_cpu.data<int32_t>()),
        reinterpret_cast<int32_t*>(block_list_padded.data<int32_t>()),
        valid_batches_enc,
        max_blocks_each,
        max_buckets_with_context,
        max_context_len,
        block_size);

    auto block_list_hpu = custom_kernel::copy_tensor_wrapper(
        dev_ctx, block_list_padded, hpu_place);

    paddle::Tensor rope_emb_seg;
    if (max_prompt_len_without_context == max_prompt_len_with_context) {
      rope_emb_seg = paddle::experimental::slice(
          rope_emb, {2}, {0}, {max_prompt_len_without_context}, {}, {});
    } else {
      std::vector<paddle::Tensor> rope_emb_segs;
      for (auto b : valid_batches_enc) {
        int start = seq_lens_decoder_cpu.data<int>()[b];
        auto seg = paddle::experimental::slice(
            rope_emb,
            {2},
            {start},
            {start + max_prompt_len_without_context},
            {},
            {});
        rope_emb_segs.push_back(seg);
      }
      rope_emb_seg = paddle::experimental::concat(rope_emb_segs, 1);
    }
    rope_emb_seg = paddle::experimental::cast(rope_emb_seg, device_dtype);

    auto total_batch_cpu_tensor = paddle::full(
        {1}, total_batch, paddle::DataType::INT32, paddle::CPUPlace());

    auto is_prompt_cpu_tensor =
        paddle::full({1}, 1, paddle::DataType::INT32, paddle::CPUPlace());

    return {src_padded,
            rope_emb_seg,
            dummy_tensor,
            block_list_hpu,
            blk_padded_hpu,
            dummy_tensor,
            dummy_tensor,
            dummy_tensor,
            batch_ids,
            total_batch_cpu_tensor,
            is_prompt_cpu_tensor};

  } else if (dec_count > 0) {
    int total_batch = find_bucket(dec_count, batch_step_decode, max_batches_in);

    auto input_ids_column_0 =
        paddle::experimental::slice(input_ids, {1}, {0}, {1}, {}, {});
    auto input_ids_cpu = input_ids_column_0.copy_to(paddle::CPUPlace(), true);

    auto src_padded = paddle::full(
        {total_batch}, 0, paddle::DataType::INT64, paddle::CPUPlace());
    pad_fill<int64_t>(const_cast<int64_t*>(input_ids_cpu.data<int64_t>()),
                      reinterpret_cast<int64_t*>(src_padded.data<int64_t>()),
                      valid_batches_dec);

    auto seq_lens_padded = paddle::full(
        {total_batch}, 0, paddle::DataType::INT32, paddle::CPUPlace());
    pad_fill<int32_t>(
        const_cast<int32_t*>(seq_lens_decoder_cpu.data<int32_t>()),
        reinterpret_cast<int32_t*>(seq_lens_padded.data<int32_t>()),
        valid_batches_dec);

    std::shared_ptr<phi::DenseTensor> seq_lens_padded_hpu =
        std::make_shared<phi::DenseTensor>();
    seq_lens_padded_hpu->Resize(phi::make_ddim({total_batch}));
    dev_ctx->Alloc(seq_lens_padded_hpu.get(), seq_lens_decoder.dtype());

    custom_kernel::copy_tensor_wrapper(
        dev_ctx, seq_lens_padded, paddle::Tensor(seq_lens_padded_hpu));

    // get rope_emb
    auto rope_emb_seg = paddle::experimental::index_select(
        rope_emb, paddle::Tensor(seq_lens_padded_hpu), {2});
    rope_emb_seg = paddle::experimental::cast(rope_emb_seg, device_dtype);
    rope_emb_seg = paddle::experimental::squeeze_(rope_emb_seg, {1});
    // rope_emb_seg = paddle::experimental::unsqueeze_(rope_emb_seg, {2});

    // get block_indices and block_offset
    auto block_indices_padded = paddle::full(
        {total_batch}, -1, paddle::DataType::INT32, paddle::CPUPlace());
    auto block_offset_padded = paddle::full(
        {total_batch}, 0, paddle::DataType::INT32, paddle::CPUPlace());

    auto [last_block_pos, seq_lens] = index_sample(
        const_cast<int*>(seq_lens_decoder_cpu.data<int>()),
        const_cast<int*>(block_tables_cpu.data<int>()),
        reinterpret_cast<int32_t*>(block_indices_padded.data<int32_t>()),
        reinterpret_cast<int32_t*>(block_offset_padded.data<int32_t>()),
        valid_batches_dec,
        max_blocks_each,
        block_size);
    auto block_indices_hpu = custom_kernel::copy_tensor_wrapper(
        dev_ctx, block_indices_padded, hpu_place);
    auto block_offset_hpu = custom_kernel::copy_tensor_wrapper(
        dev_ctx, block_offset_padded, hpu_place);

    // get batch_ids
    auto valid_batches_tensor =
        paddle::full({static_cast<int64_t>(valid_batches_dec.size())},
                     0,
                     paddle::DataType::INT32,
                     paddle::CPUPlace());
    std::memcpy(valid_batches_tensor.data<int>(),
                valid_batches_dec.data(),
                valid_batches_dec.size() * sizeof(int));

    auto batch_ids = custom_kernel::copy_tensor_wrapper(
        dev_ctx, valid_batches_tensor, hpu_place);

    // get block_list and block_groups
    auto [block_list, block_groups] = get_block_list_groups(
        valid_batches_dec,
        const_cast<int32_t*>(block_tables_cpu.data<int32_t>()),
        last_block_pos,
        max_blocks_each);

    int block_bucket_size = round_up(block_list.size(), block_step);

    auto block_list_tensor = paddle::full(
        {block_bucket_size}, -1, paddle::DataType::INT32, paddle::CPUPlace());
    std::memcpy(block_list_tensor.data<int>(),
                block_list.data(),
                block_list.size() * sizeof(int));

    auto block_groups_tensor = paddle::full(
        {block_bucket_size}, -1, paddle::DataType::INT32, paddle::CPUPlace());
    std::memcpy(block_groups_tensor.data<int>(),
                block_groups.data(),
                block_groups.size() * sizeof(int));

    auto block_groups_hpu = custom_kernel::copy_tensor_wrapper(
        dev_ctx, block_groups_tensor, hpu_place);
    auto block_list_hpu = custom_kernel::copy_tensor_wrapper(
        dev_ctx, block_list_tensor, hpu_place);

    // get block_mapping
    auto block_mapping = paddle::full({block_bucket_size, total_batch},
                                      0,
                                      paddle::DataType::BFLOAT16,
                                      paddle::CPUPlace());

    for (int i = 0; i < static_cast<int>(block_groups.size()); ++i) {
      if (block_groups[i] >= 0) {
        block_mapping
            .data<phi::dtype::bfloat16>()[i * total_batch + block_groups[i]] =
            1;
      }
    }
    block_mapping = paddle::experimental::cast(block_mapping, device_dtype);
    auto block_mapping_hpu =
        custom_kernel::copy_tensor_wrapper(dev_ctx, block_mapping, hpu_place);

    // get attention_mask
    std::vector<int> block_usage;
    for (size_t group_idx = 0; group_idx < seq_lens.size(); ++group_idx) {
      int length = seq_lens[group_idx];
      while (length > 0) {
        if (length >= block_size) {
          block_usage.push_back(block_size);
          length -= block_size;
        } else {
          block_usage.push_back(length);
          length = 0;
        }
      }
    }
    block_usage.resize(block_bucket_size, 1);

    std::vector<phi::dtype::bfloat16> attention_mask;
    for (int usage : block_usage) {
      for (int i = 0; i < usage; ++i) {
        attention_mask.push_back(static_cast<phi::dtype::bfloat16>(0.0f));
      }
      for (int i = usage; i < block_size; ++i) {
        attention_mask.push_back(
            -std::numeric_limits<phi::dtype::bfloat16>::infinity());
      }
    }

    auto attention_mask_tensor = paddle::full({block_bucket_size, block_size},
                                              0.0f,
                                              paddle::DataType::BFLOAT16,
                                              paddle::CPUPlace());
    std::memcpy(attention_mask_tensor.data<phi::dtype::bfloat16>(),
                attention_mask.data(),
                attention_mask.size() * sizeof(phi::dtype::bfloat16));
    attention_mask_tensor =
        paddle::experimental::cast(attention_mask_tensor, device_dtype);
    auto attention_mask_hpu = custom_kernel::copy_tensor_wrapper(
        dev_ctx, attention_mask_tensor, hpu_place);

    auto total_batch_cpu_tensor = paddle::full(
        {1}, total_batch, paddle::DataType::INT32, paddle::CPUPlace());
    auto is_prompt_cpu_tensor =
        paddle::full({1}, 2, paddle::DataType::INT32, paddle::CPUPlace());

    return {src_padded,
            rope_emb_seg,
            block_groups_hpu,
            block_list_hpu,
            block_indices_hpu,
            block_offset_hpu,
            block_mapping_hpu,
            attention_mask_hpu,
            batch_ids,
            total_batch_cpu_tensor,
            is_prompt_cpu_tensor};
  }

  return {dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor,
          dummy_tensor};
}

std::vector<std::vector<int64_t>> PrepareBlockMetadataShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& rope_emb_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape) {
  return {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {1}};
}

std::vector<paddle::DataType> PrepareBlockMetadataDtype(
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& rope_emb_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    int block_size,
    std::string dtype,
    int max_num_batched_tokens) {
  return {input_ids_dtype,
          phi::StringToDataType(dtype),
          phi::DataType::INT32,
          phi::DataType::INT32,
          phi::DataType::INT32,
          phi::DataType::INT32,
          phi::StringToDataType(dtype),
          phi::StringToDataType(dtype),
          phi::DataType::INT32,
          phi::DataType::INT32,
          phi::DataType::INT32};
}

PD_BUILD_OP(prepare_block_metadata)
    .Inputs({"input_ids",
             "rope_emb",
             "block_tables",
             "seq_lens_encoder",
             "seq_lens_decoder"})
    .Outputs({"ids_remove_padding",
              "rotary_embs",
              "block_groups",
              "block_list",
              "block_indices",
              "block_offsets",
              "block_mapping",
              "attention_mask",
              "batch_ids",
              "total_batch",
              "is_prompt"})
    .Attrs({"block_size: int",
            "device_dtype: std::string",
            "max_num_batched_tokens: int"})
    .SetKernelFn(PD_KERNEL(PrepareBlockMetadata))
    .SetInferShapeFn(PD_INFER_SHAPE(PrepareBlockMetadataShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PrepareBlockMetadataDtype));
